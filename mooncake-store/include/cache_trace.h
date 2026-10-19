// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
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

struct CacheRemovalRecord {
    std::string event{"hash_removed"};
    uint64_t sequence{0};
    std::string object_key;
    std::string prefix_hash;
    std::string tenant_id;
    uint64_t lookup_hit_count{0};
    // Successful replica-list responses, not confirmed data transfers/resumes.
    uint64_t read_request_count{0};
    YLT_REFL(CacheRemovalRecord, event, sequence, object_key, prefix_hash,
             tenant_id, lookup_hit_count, read_request_count);
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
            CacheRemovalRecord record;
            record.sequence = ++sequence;
            record.object_key = key;
            const auto separator = key.rfind('@');
            if (separator != std::string::npos)
                record.prefix_hash = key.substr(separator + 1);
            record.tenant_id = tenant;
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
    bool emitted_{false};
};

}  // namespace mooncake
