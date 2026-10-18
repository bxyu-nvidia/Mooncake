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
                (entry.kind != "attention" && entry.kind != "mamba" &&
                 entry.kind != "mamba_lineage" &&
                 entry.kind != "mamba_retired") ||
                !namespaces_.emplace(entry.prefix, entry).second)
                throw std::invalid_argument(
                    "Invalid or duplicate cache namespace");
            if (entry.kind == "mamba") ++expected_mamba_[entry.scope];
        }
        for (const auto& entry : namespaces) {
            if ((entry.kind == "mamba_lineage" ||
                 entry.kind == "mamba_retired") &&
                expected_mamba_.count(entry.scope) == 0)
                throw std::invalid_argument(
                    "Mamba markers require a payload scope");
        }
        for (const auto* kind : {"attention", "mamba"}) {
            for (const auto* status : {"ready", "pending"}) {
                counts_[std::string(kind) + "." + status + "_objects"] = 0;
                counts_[std::string(kind) + "." + status + "_bytes"] = 0;
            }
        }
    }

    // Visit each logical key exactly once, irrespective of replica count.
    void Add(const std::string& key, uint64_t size, bool ready,
             bool retired = false) {
        auto split = key.rfind('@');
        if (split == std::string::npos || split + 1 == key.size()) return;
        auto it = namespaces_.find(key.substr(0, split));
        if (it == namespaces_.end()) return;
        const auto& entry = it->second;
        const auto hash = key.substr(split + 1);
        if (entry.kind == "mamba_lineage" || entry.kind == "mamba_retired") {
            if (ready) {
                auto& prefix = mamba_[entry.scope][hash];
                if (entry.kind == "mamba_retired")
                    prefix.retired = true;
                else
                    prefix.lineage = true;
            }
            return;
        }
        const auto metric = entry.kind + (ready ? ".ready" : ".pending");
        ++counts_[metric + "_objects"];
        counts_[metric + "_bytes"] += size;
        if (entry.kind == "mamba") {
            auto& prefix = mamba_[entry.scope][hash];
            prefix.retired = prefix.retired || retired;
            if (ready) {
                ++prefix.ready_objects;
                prefix.ready_bytes += size;
            } else {
                ++prefix.pending_objects;
                prefix.pending_bytes += size;
            }
        } else if (ready) {
            attention_[entry.scope].insert(hash);
        }
    }

    CacheInventorySnapshot Finish() const {
        CacheInventorySnapshot result;
        result.counts = counts_;
        auto& complete = result.counts["mamba.complete_prefixes"];
        auto& incomplete = result.counts["mamba.incomplete_prefixes"];
        struct Category {
            const char* name;
            uint64_t complete{0};
            uint64_t incomplete{0};
            MambaPrefix payload{};
        };
        Category categories[]{{"anchor_or_unknown"}, {"tip"}, {"retired"}};
        for (const auto& [scope, hashes] : mamba_) {
            for (const auto& [hash, prefix] : hashes) {
                // Markers can outlive their payloads. They are not checkpoints.
                if (prefix.ready_objects + prefix.pending_objects == 0)
                    continue;
                const bool is_complete =
                    prefix.ready_objects == expected_mamba_.at(scope);
                if (is_complete)
                    ++complete;
                else
                    ++incomplete;
                // Absence of lineage is ambiguous: genuine root, lost marker,
                // untracked save, or retirement disabled. Never call it a known
                // root.
                auto& category = prefix.retired   ? categories[2]
                                 : prefix.lineage ? categories[1]
                                                  : categories[0];
                if (is_complete)
                    ++category.complete;
                else
                    ++category.incomplete;
                category.payload.ready_objects += prefix.ready_objects;
                category.payload.ready_bytes += prefix.ready_bytes;
                category.payload.pending_objects += prefix.pending_objects;
                category.payload.pending_bytes += prefix.pending_bytes;
            }
        }
        // Build metric names once per category, not once per cached prefix.
        for (const auto& category : categories) {
            const auto metric = std::string("mamba.") + category.name + ".";
            result.counts[metric + "complete_prefixes"] = category.complete;
            result.counts[metric + "incomplete_prefixes"] = category.incomplete;
            result.counts[metric + "ready_objects"] =
                category.payload.ready_objects;
            result.counts[metric + "ready_bytes"] =
                category.payload.ready_bytes;
            result.counts[metric + "pending_objects"] =
                category.payload.pending_objects;
            result.counts[metric + "pending_bytes"] =
                category.payload.pending_bytes;
        }
        auto& attention = result.counts["attention.distinct_prefixes"];
        for (const auto& [scope, hashes] : attention_)
            attention += hashes.size();
        return result;
    }

   private:
    std::unordered_map<std::string, CacheInventoryNamespace> namespaces_;
    std::unordered_map<std::string, size_t> expected_mamba_;
    struct MambaPrefix {
        uint64_t ready_objects{0};
        uint64_t ready_bytes{0};
        uint64_t pending_objects{0};
        uint64_t pending_bytes{0};
        bool lineage{false};
        bool retired{false};
    };
    std::unordered_map<std::string,
                       std::unordered_map<std::string, MambaPrefix>>
        mamba_;
    std::unordered_map<std::string, std::unordered_set<std::string>> attention_;
    std::map<std::string, uint64_t> counts_;
};
}  // namespace mooncake
