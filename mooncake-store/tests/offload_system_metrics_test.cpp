// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
#include "offload_system_metrics.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <future>

namespace mooncake::test {
namespace fs = std::filesystem;

class OffloadSystemMetricsTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto name =
            (fs::temp_directory_path() / "offload-system-XXXXXX").string();
        ASSERT_NE(::mkdtemp(name.data()), nullptr);
        root = name;
        Write("proc/meminfo",
              "MemAvailable: 4096 kB\nDirty: 2048 kB\nWriteback: 512 "
              "kB\nWritebackTmp: 8 kB\nCached: 8192 kB\nHugePages_Total: 0\n");
        Write("proc/pressure/io",
              "some avg10=12.5 avg60=6.25 avg300=1.5 total=1000000\nfull "
              "avg10=2.5 avg60=1.25 avg300=0.5 total=250000\n");
        Write("sys/devices/md0/stat",
              "10 2 200 30 20 3 400 80 2 90 150 0 0 0 0 0 0\n");
        Write("sys/devices/nvme0n1/stat", "5 1 100 15 10 1 200 40 1 45 75\n");
        fs::create_directories(root / "sys/devices/md0/slaves");
        fs::create_directory_symlink(root / "sys/devices/nvme0n1",
                                     root / "sys/devices/md0/slaves/nvme0n1");
        struct stat info{};
        ASSERT_EQ(::stat(root.c_str(), &info), 0);
        device_link = root / "sys/dev/block" /
                      (std::to_string(major(info.st_dev)) + ":" +
                       std::to_string(minor(info.st_dev)));
        fs::create_directories(device_link.parent_path());
        fs::create_directory_symlink(root / "sys/devices/md0", device_link);
        metrics =
            std::make_unique<OffloadSystemMetrics>(root / "proc", root / "sys");
        metrics->SetStoragePath(root.string());
    }

    void TearDown() override { fs::remove_all(root); }

    void Write(const fs::path& path, const std::string& contents) {
        fs::create_directories((root / path).parent_path());
        std::ofstream file(root / path);
        file << contents;
        file.close();
        ASSERT_TRUE(file.good());
    }

    std::string Sample() {
        std::string out;
        metrics->Serialize(out);
        return out;
    }

    void Has(const std::string& out, const std::string& sample) {
        EXPECT_NE(out.find("mooncake_offload_" + sample + "\n"),
                  std::string::npos)
            << sample << "\n"
            << out;
    }

    fs::path root, device_link;
    std::unique_ptr<OffloadSystemMetrics> metrics;
};

TEST_F(OffloadSystemMetricsTest, ExportsHostMemoryPressureAndDeviceCounters) {
    const auto out = Sample();
    Has(out, "host_memory_dirty_bytes 2097152");
    Has(out, "host_memory_writeback_bytes 524288");
    Has(out, "host_memory_available_bytes 4194304");
    Has(out, "host_memory_cached_bytes 8388608");
    Has(out, "host_memory_writeback_tmp_bytes 8192");
    Has(out, "host_meminfo_available 1");
    Has(out, "host_io_pressure_available 1");
    Has(out, "host_io_pressure_some_avg10_percent 12.5");
    Has(out, "host_io_pressure_full_us_total 250000");
    Has(out, "block_device_discovery_available 1");
    const std::string md = "{device=\"md0\",layer=\"filesystem\"}";
    Has(out, "block_write_bytes_total" + md + " 204800");
    Has(out, "block_write_duration_us_total" + md + " 80000");
    Has(out, "block_write_operations_total" + md + " 20");
    Has(out, "block_read_bytes_total" + md + " 102400");
    Has(out, "block_inflight_operations" + md + " 2");
    Has(out, "block_busy_us_total" + md + " 90000");
    Has(out, "block_weighted_io_us_total" + md + " 150000");
    Has(out,
        "block_write_bytes_total{device=\"nvme0n1\",layer=\"backing\"} 102400");
    const auto type = "# TYPE mooncake_offload_block_write_bytes_total counter";
    ASSERT_NE(out.find(type), std::string::npos);
    EXPECT_EQ(out.find(type, out.find(type) + 1), std::string::npos);
    EXPECT_EQ(out.find("histogram"), std::string::npos);
    EXPECT_EQ(out.find("summary"), std::string::npos);
    EXPECT_EQ(Sample(), out);  // Scrapes must not accumulate counters again.
    Write("sys/devices/md0/stat", "0 0 0 0 1 0 1 1 0 0 0\n");
    Has(Sample(), "block_write_bytes_total" + md + " 512");  // Expose resets.
}

TEST_F(OffloadSystemMetricsTest,
       ReportsUnavailableSourcesWithoutFakeZerosOrStaleValues) {
    ASSERT_FALSE(Sample().empty());
    fs::remove(root / "proc/pressure/io");
    Write("proc/meminfo", "Dirty: broken kB\nWriteback: 1 kB\n");
    Write("sys/devices/md0/stat", "0 0 0\n");
    auto out = Sample();
    Has(out, "host_meminfo_available 0");
    Has(out, "host_io_pressure_available 0");
    Has(out, "block_stats_available{device=\"md0\",layer=\"filesystem\"} 0");
    EXPECT_EQ(out.find("host_memory_dirty_bytes"), std::string::npos);
    EXPECT_EQ(out.find("host_io_pressure_some_us_total"), std::string::npos);
    EXPECT_EQ(out.find("block_write_bytes_total{device=\"md0\""),
              std::string::npos);
    Has(out,
        "block_write_bytes_total{device=\"nvme0n1\",layer=\"backing\"} 102400");
    fs::remove(device_link);
    Has(Sample(), "block_device_discovery_available 0");
    metrics->SetStoragePath((root / "missing").string());
    Has(Sample(), "block_device_discovery_available 0");
}

TEST_F(OffloadSystemMetricsTest,
       HandlesPartitionsAndDeduplicatesCyclicDeviceLinks) {
    Write("sys/devices/md0/md0p1/stat", "1 0 2 3 4 0 5 6 0 7 8\n");
    Write("sys/devices/md0/md0p1/partition", "1\n");
    fs::remove(device_link);
    fs::create_directory_symlink(root / "sys/devices/md0/md0p1", device_link);
    fs::create_directory_symlink(root / "sys/devices/md0",
                                 root / "sys/devices/md0/slaves/cycle");
    const auto out = Sample();
    Has(out, "block_device_discovery_available 1");
    Has(out,
        "block_write_bytes_total{device=\"md0p1\",layer=\"filesystem\"} 2560");
    Has(out,
        "block_write_bytes_total{device=\"md0\",layer=\"backing\"} 204800");
    const std::string leaf =
        "block_write_bytes_total{device=\"nvme0n1\",layer=\"backing\"}";
    ASSERT_NE(out.find(leaf), std::string::npos);
    EXPECT_EQ(out.find(leaf, out.find(leaf) + 1), std::string::npos);
}

TEST_F(OffloadSystemMetricsTest,
       RejectsMalformedPressureAndNegativeDeviceCounters) {
    Write("proc/pressure/io",
          "some avg10=oops avg60=1 avg300=2 total=3\nfull avg10=101 avg60=1 "
          "avg300=2 total=-1\n");
    Write("sys/devices/md0/stat", "1 0 2 3 -4 0 5 6 0 7 8\n");
    const auto out = Sample();
    Has(out, "host_io_pressure_available 0");
    EXPECT_EQ(out.find("host_io_pressure_some_avg10"), std::string::npos);
    Has(out, "block_stats_available{device=\"md0\",layer=\"filesystem\"} 0");
}

TEST_F(OffloadSystemMetricsTest,
       SupportsConcurrentScrapesAndNoStorageConfiguration) {
    OffloadSystemMetrics unconfigured(root / "proc", root / "sys");
    std::string out;
    unconfigured.Serialize(out);
    EXPECT_TRUE(out.empty());
    const auto expected = Sample();
    std::vector<std::future<std::string>> futures;
    for (int i = 0; i < 8; ++i)
        futures.push_back(
            std::async(std::launch::async, [&] { return Sample(); }));
    for (auto& future : futures) EXPECT_EQ(future.get(), expected);
}

}  // namespace mooncake::test
