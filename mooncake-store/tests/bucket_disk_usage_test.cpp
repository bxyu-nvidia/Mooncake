// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
#include "bucket_disk_usage.h"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <future>
#include <fstream>
#include <thread>

namespace mooncake::test {
namespace fs = std::filesystem;
using namespace std::chrono_literals;

class BucketDiskUsageTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto name =
            (fs::temp_directory_path() / "bucket-usage-XXXXXX").string();
        ASSERT_NE(::mkdtemp(name.data()), nullptr);
        root = name;
    }
    void TearDown() override { fs::remove_all(root); }

    void Write(const fs::path& path, size_t size = 8192) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << std::string(size, 'x');
        file.close();
        ASSERT_TRUE(file.good());
    }

    int64_t Allocated(const fs::path& path) {
        struct stat st{};
        EXPECT_EQ(::stat(path.c_str(), &st), 0);
        return st.st_blocks * 512;
    }

    fs::path root;
    BucketDiskUsage usage;
    OffloadMetrics metrics;
};

TEST_F(BucketDiskUsageTest, WritesAndDeletesDoNotForceScans) {
    ASSERT_FALSE(usage.Reconcile(root, 1h, &metrics));
    for (int i = 0; i < 25; ++i) {
        const auto path = root / (std::to_string(i) + ".bucket");
        Write(path);
        ASSERT_FALSE(usage.RefreshFile(path));
        EXPECT_EQ(usage.Bytes(), Allocated(path));
        ASSERT_FALSE(usage.Reconcile(root, 1h, &metrics));
        ASSERT_TRUE(fs::remove(path));
        ASSERT_FALSE(usage.RefreshFile(path));
        EXPECT_EQ(usage.Bytes(), 0);
        ASSERT_FALSE(usage.Reconcile(root, 1h, &metrics));
    }
    std::string serialized;
    metrics.Serialize(serialized);
    EXPECT_NE(serialized.find(
                  "mooncake_offload_disk_scan_operations_total 1.000000\n"),
              std::string::npos);
}

TEST_F(BucketDiskUsageTest, CountsAllocatedBlocksIncludingPartialFiles) {
    const auto path = root / "partial.bucket";
    ASSERT_FALSE(usage.Reconcile(root, 1h));
    Write(path, 1);
    ASSERT_FALSE(usage.RefreshFile(path));
    EXPECT_EQ(usage.Bytes(), Allocated(path));
    EXPECT_GT(*usage.Bytes(), 1);
    // A partial write that leaves a longer file must remain fully charged.
    Write(path, 32769);
    ASSERT_FALSE(usage.RefreshFile(path));
    EXPECT_EQ(usage.Bytes(), Allocated(path));
    Write(path, 1);
    ASSERT_FALSE(usage.RefreshFile(path));
    EXPECT_EQ(usage.Bytes(), Allocated(path));
}

TEST_F(BucketDiskUsageTest, PeriodicScanDiscoversUntrackedNestedFiles) {
    ASSERT_FALSE(usage.Reconcile(root, 1h));
    fs::create_directory(root / "nested");
    const auto path = root / "nested" / "orphan";
    Write(path);
    ASSERT_FALSE(usage.Reconcile(root, 1h));
    EXPECT_EQ(usage.Bytes(), 0);
    ASSERT_FALSE(usage.Reconcile(root, 0ms));
    EXPECT_EQ(usage.Bytes(), Allocated(path) + Allocated(root / "nested"));
}

TEST_F(BucketDiskUsageTest, SparseFilesUseAllocatedRatherThanApparentSize) {
    const auto path = root / "sparse";
    int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::ftruncate(fd, 1024 * 1024), 0);
    ASSERT_EQ(::close(fd), 0);
    ASSERT_FALSE(usage.Reconcile(root, 0ms));
    EXPECT_EQ(usage.Bytes(), Allocated(path));
}

TEST_F(BucketDiskUsageTest, FailedDeletionKeepsAllocation) {
    fs::create_directory(root / "bucket");
    Write(root / "bucket" / "child");
    ASSERT_FALSE(usage.Reconcile(root, 1h));
    const auto before = usage.Bytes();
    std::error_code error;
    EXPECT_FALSE(fs::remove(root / "bucket", error));
    ASSERT_TRUE(error);
    ASSERT_FALSE(usage.RefreshFile(root / "bucket"));
    EXPECT_EQ(usage.Bytes(), before);
}

TEST_F(BucketDiskUsageTest, FailedScanAndStatFailClosedUntilReconciliation) {
    Write(root / "data");
    ASSERT_FALSE(usage.Reconcile(root, 1h));
    EXPECT_TRUE(usage.RefreshFile(root / "data" / "not-a-directory"));
    EXPECT_FALSE(usage.Bytes());
    EXPECT_TRUE(usage.Reconcile(root / "missing", 0ms));
    EXPECT_FALSE(usage.Bytes());
    ASSERT_FALSE(usage.Reconcile(root, 0ms));
    EXPECT_EQ(usage.Bytes(), Allocated(root / "data"));
}

TEST_F(BucketDiskUsageTest, ScanCannotOverwriteConcurrentFileUpdates) {
    const auto path = root / "data";
    const auto created = root / "created";
    Write(path);
    ASSERT_FALSE(usage.Reconcile(root, 1h));
    std::promise<void> scanned, resume;
    auto resume_future = resume.get_future();
    bool visited = false;
    usage.SetScanHookForTest([&](const std::string& file) {
        if (file != path || visited) return;
        visited = true;
        scanned.set_value();
        resume_future.wait();
    });
    auto scan = std::async(std::launch::async, [&] {
        return usage.Reconcile(root, 0ms, &metrics);
    });
    if (scanned.get_future().wait_for(5s) != std::future_status::ready) {
        resume.set_value();
        ADD_FAILURE() << "Reconciliation did not reach the file";
        return;
    }
    // A second reconciliation does not start another scan or block updates.
    EXPECT_FALSE(usage.Reconcile(root, 0ms, &metrics));
    Write(path, 32768);
    EXPECT_FALSE(usage.RefreshFile(path));
    Write(created);
    EXPECT_FALSE(usage.RefreshFile(created));
    const auto expected = Allocated(path) + Allocated(created);
    EXPECT_EQ(usage.Bytes(), expected);
    resume.set_value();
    EXPECT_FALSE(scan.get());
    EXPECT_EQ(usage.Bytes(), expected);
    std::string serialized;
    metrics.Serialize(serialized);
    EXPECT_NE(serialized.find(
                  "mooncake_offload_disk_scan_operations_total 1.000000\n"),
              std::string::npos);
}

TEST_F(BucketDiskUsageTest, ScanCannotResurrectDeletedFile) {
    const auto path = root / "data";
    Write(path);
    ASSERT_FALSE(usage.Reconcile(root, 1h));
    usage.SetScanHookForTest([&](const std::string& file) {
        if (file == path) {
            EXPECT_TRUE(fs::remove(path));
            EXPECT_FALSE(usage.RefreshFile(path));
        }
    });
    ASSERT_FALSE(usage.Reconcile(root, 0ms));
    EXPECT_EQ(usage.Bytes(), 0);
}

TEST_F(BucketDiskUsageTest, ScanCannotClearConcurrentStatFailure) {
    const auto path = root / "data";
    Write(path);
    ASSERT_FALSE(usage.Reconcile(root, 1h));
    usage.SetScanHookForTest([&](const std::string& file) {
        EXPECT_TRUE(usage.RefreshFile(file + "/invalid"));
    });
    EXPECT_TRUE(usage.Reconcile(root, 0ms));
    EXPECT_FALSE(usage.Bytes());
    usage.SetScanHookForTest({});
    ASSERT_FALSE(usage.Reconcile(root, 0ms));
    EXPECT_EQ(usage.Bytes(), Allocated(path));
}

}  // namespace mooncake::test
