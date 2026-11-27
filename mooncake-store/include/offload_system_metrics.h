// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sys/stat.h>
#ifdef __linux__
#include <sys/sysmacros.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mooncake {

// Read only when the SSD metrics endpoint is scraped, never on the write path.
// These are host/device totals (including other processes), not per-client I/O.
// Filesystem and backing-device series overlap: do not sum across devices.
class OffloadSystemMetrics {
   public:
    explicit OffloadSystemMetrics(std::filesystem::path proc_root = "/proc",
                                  std::filesystem::path sys_root = "/sys")
        : proc_root_(std::move(proc_root)), sys_root_(std::move(sys_root)) {}

    void SetStoragePath(const std::string& path) {
        std::lock_guard lock(mutex_);
        storage_path_ = path;
    }

    void Serialize(std::string& out) const {
        std::string storage_path;
        {
            std::lock_guard lock(mutex_);
            storage_path = storage_path_;
        }
        if (storage_path.empty()) return;
        Emitter metrics;
        ReadMemory(metrics);
        ReadPressure(metrics);
        ReadDevices(storage_path, metrics);
        out += metrics.stream.str();
    }

   private:
    struct Emitter {
        std::ostringstream stream;
        std::set<std::string> declared;

        Emitter() {
            stream.imbue(std::locale::classic());
            stream << std::setprecision(17);
        }

        void Add(const std::string& name, const char* type, double value,
                 const std::string& labels = "") {
            const std::string metric = "mooncake_offload_" + name;
            if (declared.insert(metric).second)
                stream << "# TYPE " << metric << ' ' << type << '\n';
            stream << metric << labels << ' ' << value << '\n';
        }
    };

    static std::string Escape(const std::string& value) {
        std::string result;
        for (char c : value) {
            if (c == '\n') {
                result += "\\n";
            } else {
                if (c == '\\' || c == '"') result += '\\';
                result += c;
            }
        }
        return result;
    }

    void ReadMemory(Emitter& metrics) const {
        const std::map<std::string, std::string> fields = {
            {"Dirty:", "dirty"},
            {"Writeback:", "writeback"},
            {"WritebackTmp:", "writeback_tmp"},
            {"MemAvailable:", "available"},
            {"Cached:", "cached"}};
        std::ifstream file(proc_root_ / "meminfo");
        std::map<std::string, double> values;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream row(line);
            std::string key, unit;
            double value;
            if (row >> key >> value >> unit && fields.count(key) &&
                unit == "kB" && std::isfinite(value) && value >= 0)
                values[fields.at(key)] = value * 1024;
        }
        const bool available =
            values.count("dirty") && values.count("writeback");
        metrics.Add("host_meminfo_available", "gauge", available);
        if (available)
            for (const auto& [key, value] : values)
                metrics.Add("host_memory_" + key + "_bytes", "gauge", value);
    }

    void ReadPressure(Emitter& metrics) const {
        std::ifstream file(proc_root_ / "pressure/io");
        std::set<std::string> emitted;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream row(line);
            std::string kind, token;
            row >> kind;
            if ((kind != "some" && kind != "full") || emitted.count(kind))
                continue;
            std::map<std::string, double> values;
            while (row >> token) {
                const auto equals = token.find('=');
                if (equals == std::string::npos) continue;
                std::istringstream number(token.substr(equals + 1));
                double value;
                if (number >> value && number.eof() && std::isfinite(value) &&
                    value >= 0)
                    values[token.substr(0, equals)] = value;
            }
            if (!values.count("total") || !values.count("avg10") ||
                !values.count("avg60") || !values.count("avg300") ||
                values.at("avg10") > 100 || values.at("avg60") > 100 ||
                values.at("avg300") > 100)
                continue;
            emitted.insert(kind);
            for (const auto* window : {"avg10", "avg60", "avg300"})
                metrics.Add(
                    "host_io_pressure_" + kind + "_" + window + "_percent",
                    "gauge", values.at(window));
            metrics.Add("host_io_pressure_" + kind + "_us_total", "counter",
                        values.at("total"));
        }
        metrics.Add("host_io_pressure_available", "gauge", emitted.size() == 2);
    }

    static void ReadDevice(const std::filesystem::path& device, bool filesystem,
                           Emitter& metrics) {
        const auto labels = "{device=\"" + Escape(device.filename().string()) +
                            "\",layer=\"" +
                            (filesystem ? "filesystem" : "backing") + "\"}";
        std::ifstream file(device / "stat");
        std::array<uint64_t, 11> fields{};
        bool available = true;
        for (auto& field : fields) {
            // Reject truncated or malformed files; never turn read failures
            // into zero counters (which would look like a device reset).
            file >> std::ws;
            if (file.peek() == '-' || !(file >> field)) {
                available = false;
                break;
            }
        }
        metrics.Add("block_stats_available", "gauge", available, labels);
        if (!available) return;
        // Linux block stat sectors are always 512 bytes, including 4K disks.
        // Durations are summed request lifetimes, not wall time or fsync time.
        for (const auto& [direction, offset] :
             {std::pair{"read", 0}, std::pair{"write", 4}}) {
            const auto prefix = std::string("block_") + direction;
            metrics.Add(prefix + "_operations_total", "counter", fields[offset],
                        labels);
            metrics.Add(prefix + "_bytes_total", "counter",
                        fields[offset + 2] * 512.0, labels);
            metrics.Add(prefix + "_duration_us_total", "counter",
                        fields[offset + 3] * 1000.0, labels);
        }
        metrics.Add("block_inflight_operations", "gauge", fields[8], labels);
        metrics.Add("block_busy_us_total", "counter", fields[9] * 1000.0,
                    labels);
        metrics.Add("block_weighted_io_us_total", "counter",
                    fields[10] * 1000.0, labels);
    }

    void ReadDevices(const std::string& storage_path, Emitter& metrics) const {
        namespace fs = std::filesystem;
        struct stat info{};
        std::vector<fs::path> pending;
        if (::stat(storage_path.c_str(), &info) == 0) {
            const auto id = std::to_string(major(info.st_dev)) + ":" +
                            std::to_string(minor(info.st_dev));
            pending.push_back(sys_root_ / "dev/block" / id);
        }
        bool complete = !pending.empty();
        std::set<fs::path> visited;
        // sysfs links are followed only through partition parents and slaves,
        // with a bound and deduplication for stacked RAID/device-mapper
        // layouts.
        for (size_t i = 0; i < pending.size() && i < 64; ++i) {
            std::error_code error;
            const auto device = fs::canonical(pending[i], error);
            if (error) {
                complete = false;
                continue;
            }
            if (!visited.insert(device).second) continue;
            ReadDevice(device, i == 0, metrics);
            if (fs::exists(device / "partition", error)) {
                if (pending.size() < 64)
                    pending.push_back(device.parent_path());
                else
                    complete = false;
            }
            if (error) complete = false;
            if (fs::exists(device / "slaves", error)) {
                fs::directory_iterator it(device / "slaves", error), end;
                while (!error && it != end && pending.size() < 64) {
                    pending.push_back(it->path());
                    it.increment(error);
                }
                if (it != end) complete = false;
            }
            if (error) complete = false;
        }
        metrics.Add("block_device_discovery_available", "gauge",
                    complete && !visited.empty());
    }

    std::filesystem::path proc_root_;
    std::filesystem::path sys_root_;
    mutable std::mutex mutex_;
    std::string storage_path_;
};

}  // namespace mooncake
