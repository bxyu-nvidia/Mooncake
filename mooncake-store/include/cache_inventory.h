// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mooncake {

// Explicit expected namespaces, supplied by the consumer's actual cache layout.
// Missing shards must not be inferred away from the set of currently live keys.
struct CacheInventoryNamespace {
    std::string scope;
    std::string prefix;
    std::string kind;
};

struct CacheInventorySnapshot {
    std::map<std::string, uint64_t> counts;
    double snapshot_timestamp_seconds{0};
    double scan_seconds{0};
};

class CacheInventoryAccumulator {
   public:
    explicit CacheInventoryAccumulator(
        const std::vector<CacheInventoryNamespace>& namespaces) {
        if (namespaces.empty() || namespaces.size() > 4096)
            throw std::invalid_argument("Expected 1..4096 cache namespaces");
        for (const auto& entry : namespaces) {
            if (entry.prefix.empty() || entry.prefix.size() > 4096 ||
                entry.scope.empty() || entry.scope.size() > 4096 ||
                (entry.kind != "attention" && entry.kind != "mamba") ||
                !namespaces_.emplace(entry.prefix, entry).second)
                throw std::invalid_argument(
                    "Invalid or duplicate cache namespace");
            if (entry.kind == "mamba") ++expected_mamba_[entry.scope];
        }
        for (const auto* kind : {"attention", "mamba"}) {
            for (const auto* status : {"ready", "pending"}) {
                counts_[std::string(kind) + "." + status + "_objects"] = 0;
                counts_[std::string(kind) + "." + status + "_bytes"] = 0;
            }
        }
    }

    // Visit each logical key exactly once, irrespective of replica count.
    void Add(const std::string& key, uint64_t size, bool ready) {
        auto split = key.rfind('@');
        if (split == std::string::npos || split + 1 == key.size()) return;
        auto it = namespaces_.find(key.substr(0, split));
        if (it == namespaces_.end()) return;
        const auto& entry = it->second;
        const auto hash = key.substr(split + 1);
        const auto metric = entry.kind + (ready ? ".ready" : ".pending");
        ++counts_[metric + "_objects"];
        counts_[metric + "_bytes"] += size;
        if (entry.kind == "mamba") {
            mamba_[entry.scope][hash] += ready ? 1 : 0;
        } else if (ready) {
            attention_[entry.scope].insert(hash);
        }
    }

    CacheInventorySnapshot Finish() const {
        CacheInventorySnapshot result;
        result.counts = counts_;
        auto& complete = result.counts["mamba.complete_prefixes"];
        auto& incomplete = result.counts["mamba.incomplete_prefixes"];
        for (const auto& [scope, hashes] : mamba_) {
            for (const auto& [hash, present] : hashes) {
                if (present == expected_mamba_.at(scope))
                    ++complete;
                else
                    ++incomplete;
            }
        }
        auto& attention = result.counts["attention.distinct_prefixes"];
        for (const auto& [scope, hashes] : attention_)
            attention += hashes.size();
        return result;
    }

   private:
    std::unordered_map<std::string, CacheInventoryNamespace> namespaces_;
    std::unordered_map<std::string, size_t> expected_mamba_;
    std::unordered_map<std::string, std::unordered_map<std::string, size_t>>
        mamba_;
    std::unordered_map<std::string, std::unordered_set<std::string>> attention_;
    std::map<std::string, uint64_t> counts_;
};
}  // namespace mooncake
