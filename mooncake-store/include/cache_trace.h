// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <unistd.h>
#include <glog/logging.h>
#include <ylt/struct_json/json_writer.h>

namespace mooncake {

inline bool CacheTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("MOONCAKE_CACHE_TRACE");
        return value && std::string(value) == "1";
    }();
    return enabled;
}

inline int64_t CacheTraceNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

struct CacheRemovalRecord {
    int schema_version{2};
    std::string event{"hash_removed"};
    std::string medium{"Mooncake"};
    std::string instance_id;
    std::string run_id;
    uint64_t sequence{0};
    std::string object_key;
    std::string prefix_hash;
    std::string tenant_id;
    int64_t created_ns{0};
    int64_t removed_ns{0};
    int64_t lifetime_ns{0};
    uint64_t lookup_hit_count{0};
    // Successful replica-list responses, not confirmed data transfers/resumes.
    uint64_t read_request_count{0};
    YLT_REFL(CacheRemovalRecord, schema_version, event, medium, instance_id,
             run_id, sequence, object_key, prefix_hash, tenant_id, created_ns,
             removed_ns, lifetime_ns, lookup_hit_count, read_request_count);
};

// One instance per object key, shared by all its replicas. Readers hold shared
// metadata locks, so the counters must be atomic. Only object removal emits.
class CacheObjectTrace {
   public:
    std::atomic<uint64_t> lookup_hits{0};
    std::atomic<uint64_t> read_requests{0};

    void Removed(const std::string& key, const std::string& tenant) noexcept {
        if (emitted_) return;
        emitted_ = true;
        try {
            static std::atomic<uint64_t> sequence{0};
            static const std::string instance_id = [] {
                char host[256]{};
                gethostname(host, sizeof(host) - 1);
                return std::string(host) + ":" + std::to_string(getpid()) +
                       ":" + std::to_string(CacheTraceNowNs());
            }();
            CacheRemovalRecord record;
            record.instance_id = instance_id;
            record.sequence = ++sequence;
            if (const char* run = std::getenv("VLLM_MAMBA_CACHE_TRACE_RUN_ID"))
                record.run_id = run;
            record.object_key = key;
            const auto separator = key.rfind('@');
            if (separator != std::string::npos)
                record.prefix_hash = key.substr(separator + 1);
            record.tenant_id = tenant;
            record.created_ns = created_ns_;
            record.removed_ns = CacheTraceNowNs();
            record.lifetime_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - birth_)
                    .count();
            record.lookup_hit_count =
                lookup_hits.load(std::memory_order_relaxed);
            record.read_request_count =
                read_requests.load(std::memory_order_relaxed);
            std::string json;
            struct_json::to_json(record, json);
            LOG(INFO) << "MAMBA_CACHE_TRACE " << json;
        } catch (const std::exception& e) {
            LOG(WARNING) << "Cache trace serialization failed: " << e.what();
        }
    }

   private:
    const int64_t created_ns_{CacheTraceNowNs()};
    const std::chrono::steady_clock::time_point birth_{
        std::chrono::steady_clock::now()};
    bool emitted_{false};
};

}  // namespace mooncake
