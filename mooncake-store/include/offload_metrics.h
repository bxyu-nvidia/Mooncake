// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace mooncake {

// Fixed-cardinality, histogram-free diagnostics. Timings include failed calls;
// consumers derive interval means from duration_us_total / operations_total.
class OffloadMetrics {
   public:
    static int64_t NowUs() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
#define OFFLOAD_COUNTERS(X)                                                \
    X(eviction_passes)                                                     \
    X(eviction_watermark_passes) X(eviction_write_checks)                  \
        X(eviction_stop_target) X(eviction_stop_bucket_limit)              \
            X(eviction_stop_no_candidates) X(eviction_errors)              \
                X(eviction_selected_buckets) X(eviction_deleted_buckets)   \
                    X(eviction_deleted_bytes) X(eviction_cleanup_failures) \
                        X(read_buffer_allocation_failures)                 \
                            X(read_buffer_requested_bytes)                 \
                                X(read_buffer_requests)                    \
                                    X(offload_received_objects)            \
                                        X(offload_received_bytes)
#define OFFLOAD_GAUGES(X)                                                 \
    X(disk_logical_bytes)                                                 \
    X(disk_quota_bytes) X(disk_high_watermark_bytes)                      \
        X(disk_low_watermark_bytes) X(disk_above_low_bytes)               \
            X(eviction_before_bytes) X(eviction_after_bytes)              \
                X(read_buffer_capacity_bytes) X(read_buffer_in_use_bytes) \
                    X(read_buffer_batches) X(read_buffer_peak_bytes)      \
                        X(offload_batch_remaining_bytes)                  \
                            X(offload_batch_remaining_objects)
#define OFFLOAD_PHASES(X)                                                 \
    X(eviction_trigger_delay)                                             \
    X(eviction_pass) X(eviction_prepare) X(eviction_lock_wait)            \
        X(eviction_lock_hold) X(eviction_select) X(disk_scan)             \
            X(eviction_notify) X(eviction_reader_wait) X(eviction_delete) \
                X(heartbeat) X(heartbeat_rpc) X(offload_batch)            \
                    X(offload_staging) X(offload_write) X(offload_ack)    \
                        X(read_buffer_lifetime)
#define ENUM_VALUE(name) name,
    enum class Counter { OFFLOAD_COUNTERS(ENUM_VALUE) size };
    enum class Gauge { OFFLOAD_GAUGES(ENUM_VALUE) size };
    enum class Phase { OFFLOAD_PHASES(ENUM_VALUE) size };
#undef ENUM_VALUE
    void Add(Counter key, int64_t n = 1) {
        counters_[size_t(key)].fetch_add(n, std::memory_order_relaxed);
    }
    void Set(Gauge key, int64_t n) {
        gauges_[size_t(key)].store(n, std::memory_order_relaxed);
    }
    int64_t AddGauge(Gauge key, int64_t n) {
        return gauges_[size_t(key)].fetch_add(n, std::memory_order_relaxed) + n;
    }
    int64_t Get(Gauge key) const {
        return gauges_[size_t(key)].load(std::memory_order_relaxed);
    }
    int64_t Get(Counter key) const {
        return counters_[size_t(key)].load(std::memory_order_relaxed);
    }
    void Max(Gauge key, int64_t n) { UpdateMax(gauges_[size_t(key)], n); }
    void Observe(Phase key, int64_t us) {
        auto& timing = phases_[size_t(key)];
        timing.us.fetch_add(std::max<int64_t>(0, us),
                            std::memory_order_relaxed);
        timing.operations.fetch_add(1, std::memory_order_relaxed);
        UpdateMax(timing.max_us, us);
    }
    // Called while the backend metadata lock is held. This only records
    // observations; it does not latch or change the eviction policy.
    void Usage(int64_t used, int64_t quota, double high, double low) {
        Set(Gauge::disk_logical_bytes, used);
        Set(Gauge::disk_quota_bytes, quota);
        Set(Gauge::disk_high_watermark_bytes, int64_t(quota * high));
        Set(Gauge::disk_low_watermark_bytes, int64_t(quota * low));
        Set(Gauge::disk_above_low_bytes,
            std::max<int64_t>(0, used - int64_t(quota * low)));
        if (quota > 0 && used > int64_t(quota * high)) {
            int64_t empty = 0;
            const auto now = NowUs();
            if (above_high_since_us.compare_exchange_strong(empty, now))
                pending_trigger_since_us.store(now);
        } else {
            above_high_since_us.store(0);
            pending_trigger_since_us.store(0);
        }
    }
    void Serialize(std::string& out) const {
#define NAME_VALUE(name) #name,
        static constexpr const char* counters[] = {
            OFFLOAD_COUNTERS(NAME_VALUE)};
        static constexpr const char* gauges[] = {OFFLOAD_GAUGES(NAME_VALUE)};
        static constexpr const char* phases[] = {OFFLOAD_PHASES(NAME_VALUE)};
#undef NAME_VALUE
        const std::string prefix = "mooncake_offload_";
        const auto metric_name = [&prefix](const char* field) {
            std::string name(field);
            if (name.rfind("offload_", 0) == 0) name.erase(0, 8);
            return prefix + name;
        };
        for (size_t i = 0; i < size_t(Counter::size); ++i)
            Emit(out, metric_name(counters[i]) + "_total", "counter",
                 counters_[i].load());
        for (size_t i = 0; i < size_t(Gauge::size); ++i)
            Emit(out, metric_name(gauges[i]), "gauge", gauges_[i].load());
        for (size_t i = 0; i < size_t(Phase::size); ++i) {
            Emit(out, metric_name(phases[i]) + "_duration_us_total", "counter",
                 phases_[i].us.load());
            Emit(out, metric_name(phases[i]) + "_operations_total", "counter",
                 phases_[i].operations.load());
            Emit(out, metric_name(phases[i]) + "_max_duration_us", "gauge",
                 phases_[i].max_us.load());
        }
        Age(out, prefix + "above_high_age_seconds", above_high_since_us.load());
        Age(out, prefix + "batch_age_seconds", batch_since_us.load());
        Age(out, prefix + "eviction_pending_age_seconds",
            pending_trigger_since_us.load());
    }
    static void Emit(std::string& out, const std::string& name,
                     const char* type, double value) {
        out += "# TYPE " + name + " " + type + "\n" + name + " " +
               std::to_string(value) + "\n";
    }
    static void Age(std::string& out, const std::string& name, int64_t since) {
        Emit(out, name, "gauge",
             since ? std::max<int64_t>(0, NowUs() - since) / 1e6 : 0);
    }
    std::atomic<int64_t> pending_trigger_since_us{0};
    std::atomic<int64_t> above_high_since_us{0};
    std::atomic<int64_t> batch_since_us{0};

   private:
    static void UpdateMax(std::atomic<int64_t>& target, int64_t value) {
        auto old = target.load(std::memory_order_relaxed);
        while (old < value && !target.compare_exchange_weak(
                                  old, value, std::memory_order_relaxed)) {
        }
    }
    struct Timing {
        std::atomic<int64_t> us{0}, operations{0}, max_us{0};
    };
    std::array<std::atomic<int64_t>, size_t(Counter::size)> counters_{};
    std::array<std::atomic<int64_t>, size_t(Gauge::size)> gauges_{};
    std::array<Timing, size_t(Phase::size)> phases_{};
#undef OFFLOAD_COUNTERS
#undef OFFLOAD_GAUGES
#undef OFFLOAD_PHASES
};

class OffloadTimer {
   public:
    OffloadTimer(OffloadMetrics* metrics, OffloadMetrics::Phase phase)
        : metrics_(metrics),
          phase_(phase),
          start_(metrics ? OffloadMetrics::NowUs() : 0) {}
    ~OffloadTimer() { Stop(); }
    OffloadTimer(const OffloadTimer&) = delete;
    OffloadTimer& operator=(const OffloadTimer&) = delete;
    void Stop() {
        if (metrics_) {
            metrics_->Observe(phase_, OffloadMetrics::NowUs() - start_);
            metrics_ = nullptr;
        }
    }

   private:
    OffloadMetrics* metrics_;
    OffloadMetrics::Phase phase_;
    int64_t start_;
};

// Master-owned task accounting. Ages use the master's monotonic clock; no
// timestamps or additional fields cross the RPC boundary. The sets contain
// timestamps only (not keys), and scraping never walks the task population.
class OffloadSourceMetrics {
   public:
    static OffloadSourceMetrics& Instance() {
        static OffloadSourceMetrics metrics;
        return metrics;
    }
    void Admit(int64_t start, int64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        queued_.insert(start);
        queued_bytes_ += bytes;
        ++admitted_;
    }
    void Dispatch(int64_t start, int64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = queued_.find(start);
        if (it == queued_.end()) return;
        queued_.erase(it);
        dispatched_.insert(start);
        queued_bytes_ -= bytes;
        dispatched_bytes_ += bytes;
        queue_us_ += std::max<int64_t>(0, OffloadMetrics::NowUs() - start);
        ++dispatches_;
    }
    void Release(int64_t start, int64_t bytes, bool dispatched) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& entries = dispatched ? dispatched_ : queued_;
        auto it = entries.find(start);
        if (it == entries.end()) return;
        entries.erase(it);
        (dispatched ? dispatched_bytes_ : queued_bytes_) -= bytes;
        source_us_ += std::max<int64_t>(0, OffloadMetrics::NowUs() - start);
        ++released_;
    }
    void Expired() { expired_.fetch_add(1, std::memory_order_relaxed); }
    void Serialize(std::string& out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string p = "master_offload_diagnostics_";
        OffloadMetrics::Emit(out, p + "queued_objects", "gauge",
                             queued_.size());
        OffloadMetrics::Emit(out, p + "queued_bytes", "gauge", queued_bytes_);
        OffloadMetrics::Emit(out, p + "dispatched_objects", "gauge",
                             dispatched_.size());
        OffloadMetrics::Emit(out, p + "dispatched_bytes", "gauge",
                             dispatched_bytes_);
        OffloadMetrics::Age(out, p + "queued_oldest_age_seconds",
                            queued_.empty() ? 0 : *queued_.begin());
        OffloadMetrics::Age(out, p + "dispatched_oldest_age_seconds",
                            dispatched_.empty() ? 0 : *dispatched_.begin());
        OffloadMetrics::Emit(out, p + "admitted_total", "counter", admitted_);
        OffloadMetrics::Emit(out, p + "released_total", "counter", released_);
        OffloadMetrics::Emit(out, p + "expired_total", "counter",
                             expired_.load());
        OffloadMetrics::Emit(out, p + "queue_duration_us_total", "counter",
                             queue_us_);
        OffloadMetrics::Emit(out, p + "queue_operations_total", "counter",
                             dispatches_);
        OffloadMetrics::Emit(out, p + "source_reference_duration_us_total",
                             "counter", source_us_);
        OffloadMetrics::Emit(out, p + "source_reference_operations_total",
                             "counter", released_);
    }

   private:
    mutable std::mutex mutex_;
    std::multiset<int64_t> queued_, dispatched_;
    int64_t queued_bytes_ = 0, dispatched_bytes_ = 0;
    int64_t admitted_ = 0, released_ = 0, dispatches_ = 0, queue_us_ = 0,
            source_us_ = 0;
    std::atomic<int64_t> expired_{0};
};
}  // namespace mooncake
