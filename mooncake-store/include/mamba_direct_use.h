// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#pragma once

#include <algorithm>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mooncake {

// Process-lifetime history, independent of object residency. The caller
// supplies tenant-scoped object keys and a stable ID for each completed
// direct-child use.
class MambaDirectUsePolicy {
   public:
    int Report(const std::string& scoped_key, const std::string& event_id,
               const std::string& key, const std::string& tenant) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto [it, inserted] = history_.try_emplace(
            scoped_key, Entry{event_id, key, tenant, false});
        auto& entry = it->second;
        if (inserted) {
            pending_.push_back(scoped_key);
        } else if (!entry.protected_state && entry.first_event != event_id) {
            entry.protected_state = true;
            entry.first_event.clear();
        }
        return entry.protected_state ? 2 : 1;
    }

    // Removal and protection are serialized: once a second report returns,
    // no pending policy removal can delete that object. A false result means
    // a lease/write/task still prevents removal; retry in a later cycle.
    template <typename Remove>
    void Drain(Remove&& remove, size_t budget = 64) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t count = std::min(budget, pending_.size());
        for (size_t i = 0; i < count; ++i) {
            auto scoped_key = std::move(pending_.front());
            pending_.pop_front();
            const auto& entry = history_.at(scoped_key);
            if (!entry.protected_state && !remove(entry.key, entry.tenant)) {
                pending_.push_back(std::move(scoped_key));
            }
        }
    }

   private:
    struct Entry {
        std::string first_event;
        std::string key;
        std::string tenant;
        bool protected_state;
    };
    std::mutex mutex_;
    std::unordered_map<std::string, Entry> history_;
    std::deque<std::string> pending_;
};

}  // namespace mooncake
