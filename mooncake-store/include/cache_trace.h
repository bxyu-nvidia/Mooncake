// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <chrono>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <unistd.h>
#include <glog/logging.h>
#include <ylt/struct_json/json_writer.h>
#include "tenant_id.h"

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

inline uint64_t CacheTraceSequence() {
    static std::atomic<uint64_t> sequence{0};
    return ++sequence;
}

// Travels with a replica through staging lists and vector moves. The object key
// joins these records to vLLM's request/transfer records without an extra RPC.
struct CacheReplicaTrace {
    int schema_version{1};
    uint64_t sequence{0};
    std::string event;
    std::string instance_id;
    std::string run_id;
    std::string object_key;
    std::string tenant_id;
    uint64_t replica_id{0};
    std::string tier;
    std::string descriptor_json;
    int64_t timestamp_ns{0};
    int64_t attached_ns{CacheTraceNowNs()};
    std::optional<int64_t> committed_ns;
    std::optional<int64_t> retired_ns;
    std::optional<int64_t> storage_reclaimed_ns;
    std::string reason;
    int64_t observed_lifetime_ns{0};
    std::chrono::steady_clock::time_point birth{
        std::chrono::steady_clock::now()};

    YLT_REFL(CacheReplicaTrace, schema_version, sequence, event, instance_id,
             run_id, object_key, tenant_id, replica_id, tier, descriptor_json,
             timestamp_ns, attached_ns, committed_ns, retired_ns,
             storage_reclaimed_ns, reason, observed_lifetime_ns);

    static const std::string& InstanceId() {
        static const std::string value = [] {
            char host[256]{};
            gethostname(host, sizeof(host) - 1);
            return std::string(host) + ":" + std::to_string(getpid()) + ":" +
                   std::to_string(CacheTraceNowNs());
        }();
        return value;
    }

    void Emit(const char* kind) noexcept {
        try {
            event = kind;
            sequence = CacheTraceSequence();
            instance_id = InstanceId();
            if (const char* value =
                    std::getenv("VLLM_MAMBA_CACHE_TRACE_RUN_ID")) {
                run_id = value;
            }
            timestamp_ns = CacheTraceNowNs();
            observed_lifetime_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - birth)
                    .count();
            std::string json;
            struct_json::to_json(*this, json);
            LOG(INFO) << "MAMBA_CACHE_TRACE " << json;
        } catch (const std::exception& e) {
            LOG(WARNING) << "Cache trace serialization failed: " << e.what();
        }
    }
};

struct CacheStorageReclaimed {
    int schema_version{1};
    uint64_t sequence{CacheTraceSequence()};
    std::string event{"mooncake_storage_reclaimed"};
    std::string instance_id{CacheReplicaTrace::InstanceId()};
    std::string run_id;
    std::string object_key;
    std::string tenant_id;
    std::string tier{"local_disk"};
    std::string path;
    int64_t storage_reclaimed_ns{CacheTraceNowNs()};
    std::string reason;
    YLT_REFL(CacheStorageReclaimed, schema_version, sequence, event,
             instance_id, run_id, object_key, tenant_id, tier, path,
             storage_reclaimed_ns, reason);
};

inline void TraceStorageReclaimed(const std::string& storage_key,
                                  const std::string& path, const char* reason) {
    if (!CacheTraceEnabled()) return;
    auto [tenant, key] = TenantId::ParseScopedKey(storage_key);
    CacheStorageReclaimed record;
    record.object_key = key;
    record.tenant_id = tenant.value();
    record.path = path;
    record.reason = reason;
    if (const char* value = std::getenv("VLLM_MAMBA_CACHE_TRACE_RUN_ID")) {
        record.run_id = value;
    }
    std::string json;
    struct_json::to_json(record, json);
    LOG(INFO) << "MAMBA_CACHE_TRACE " << json;
}

}  // namespace mooncake
