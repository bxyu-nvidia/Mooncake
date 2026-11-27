// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sys/stat.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

#include "offload_metrics.h"

namespace mooncake {

// Physical allocation of whole bucket files, including orphan files and files
// whose objects were only partially invalidated. Independent of logical quota
// accounting. Call RefreshFile after every write/unlink, including failures.
class BucketDiskUsage {
   public:
    // Return no estimate until a complete scan succeeds, or after an I/O error.
    // Callers must reject writes when physical usage cannot be established.
    std::optional<int64_t> Bytes() const {
        std::lock_guard lock(mutex_);
        return valid_ ? std::optional<int64_t>(bytes_) : std::nullopt;
    }

    int64_t FileBytes(const std::string& path) const {
        std::lock_guard lock(mutex_);
        const auto it = files_.find(path);
        return it == files_.end() ? 0 : it->second;
    }

    std::error_code RefreshFile(const std::string& path) {
        std::lock_guard lock(mutex_);
        std::error_code error;
        const auto bytes = AllocatedBytes(path, error);
        if (error) {
            valid_ = false;
            if (scanning_) scan_update_error_ = error;
            return error;
        }
        Replace(files_, bytes_, path, bytes);
        // Keep zero-byte tombstones until the scan merges its snapshot, so a
        // concurrent unlink cannot be undone by an earlier stat from that scan.
        if (scanning_) scan_updates_[path] = bytes;
        return {};
    }

    // One caller performs a due scan; other callers keep using the tracked
    // allocation plus their write reservations. No bucket metadata lock may be
    // held here. File updates continue while the directory is being traversed.
    std::error_code Reconcile(const std::string& root,
                              std::chrono::milliseconds interval,
                              OffloadMetrics* metrics = nullptr) {
        {
            std::lock_guard lock(mutex_);
            if (scanning_ ||
                (last_scan_ && Clock::now() - *last_scan_ < interval)) {
                return {};
            }
            scanning_ = true;
            scan_updates_.clear();
            scan_update_error_.clear();
        }
        OffloadTimer timer(metrics, OffloadMetrics::Phase::disk_scan);
        std::unordered_map<std::string, int64_t> snapshot;
        int64_t scanned_bytes = 0;
        std::error_code error;
        try {
            namespace fs = std::filesystem;
            fs::recursive_directory_iterator it(root, error), end;
            while (!error && it != end) {
                const auto path = it->path().string();
                const auto bytes = AllocatedBytes(path, error);
                if (error) break;
                Replace(snapshot, scanned_bytes, path, bytes);
                if (scan_hook_) scan_hook_(path);
                it.increment(error);
            }
        } catch (...) {
            // Do not strand the single-scanner guard if traversal or allocation
            // throws. The caller still receives the original exception.
            std::lock_guard lock(mutex_);
            scanning_ = false;
            valid_ = false;
            throw;
        }

        std::lock_guard lock(mutex_);
        if (!error) error = scan_update_error_;
        if (!error) {
            for (const auto& [path, bytes] : scan_updates_) {
                Replace(snapshot, scanned_bytes, path, bytes);
            }
            files_.swap(snapshot);
            bytes_ = scanned_bytes;
            valid_ = true;
        } else {
            // Never replace the estimate with a partial directory scan.
            valid_ = false;
        }
        scan_updates_.clear();
        scanning_ = false;
        last_scan_ = Clock::now();
        return error;
    }

    // Install before starting threads. Called after each stat, outside mutex_,
    // to make scan/write and scan/unlink races deterministic in unit tests.
    void SetScanHookForTest(std::function<void(const std::string&)> hook) {
        scan_hook_ = std::move(hook);
    }

   private:
    using Clock = std::chrono::steady_clock;

    static int64_t AllocatedBytes(const std::string& path,
                                  std::error_code& error) {
        struct stat st;
        if (::stat(path.c_str(), &st) == 0) {
            return static_cast<int64_t>(st.st_blocks) * 512;
        }
        if (errno != ENOENT) error = {errno, std::generic_category()};
        return 0;
    }

    static void Replace(std::unordered_map<std::string, int64_t>& files,
                        int64_t& total, const std::string& path,
                        int64_t bytes) {
        auto it = files.find(path);
        if (it != files.end()) {
            total -= it->second;
            if (bytes == 0) {
                files.erase(it);
            } else {
                it->second = bytes;
            }
        } else if (bytes != 0) {
            files.emplace(path, bytes);
        }
        total += bytes;
    }

    mutable std::mutex mutex_;
    std::unordered_map<std::string, int64_t> files_;
    int64_t bytes_{0};
    bool valid_{false};
    bool scanning_{false};
    std::optional<Clock::time_point> last_scan_;
    std::unordered_map<std::string, int64_t> scan_updates_;
    std::error_code scan_update_error_;
    std::function<void(const std::string&)> scan_hook_;
};

}  // namespace mooncake
