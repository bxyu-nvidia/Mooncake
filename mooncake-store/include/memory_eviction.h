// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <list>

namespace mooncake {

// Protected by the owning metadata shard's lock. Membership follows replica
// additions/removals, so sampling never walks disk-only object metadata.
template <typename T>
class MemoryEvictionIndex {
   public:
    class Membership {
       public:
        Membership(MemoryEvictionIndex* index, T* object)
            : index_(index), object_(object) {}
        ~Membership() { SetResident(false); }
        Membership(const Membership&) = delete;
        Membership& operator=(const Membership&) = delete;

        void SetResident(bool resident) {
            if (index_ == nullptr || resident == resident_) return;
            if (resident) {
                position_ =
                    index_->objects_.insert(index_->objects_.end(), object_);
            } else {
                index_->objects_.erase(position_);
            }
            resident_ = resident;
        }

       private:
        MemoryEvictionIndex* index_;
        T* object_;
        typename std::list<T*>::iterator position_;
        bool resident_{false};
    };

    MemoryEvictionIndex() = default;
    MemoryEvictionIndex(const MemoryEvictionIndex&) = delete;
    MemoryEvictionIndex& operator=(const MemoryEvictionIndex&) = delete;
    ~MemoryEvictionIndex() { assert(objects_.empty()); }
    size_t size() const { return objects_.size(); }

    template <typename Visitor>
    size_t Visit(size_t limit, Visitor&& visitor) {
        const size_t count = std::min(limit, objects_.size());
        for (size_t i = 0; i < count; ++i) {
            T* object = objects_.front();
            objects_.splice(objects_.end(), objects_, objects_.begin());
            visitor(*object);
        }
        return count;
    }

   private:
    std::list<T*> objects_;
};

// Allocation failures must not defeat the cooldown: an offload-pinned cache
// cannot free space just because another writer failed to allocate.
class MemoryEvictionBackoff {
   public:
    using Clock = std::chrono::steady_clock;
    bool Ready(Clock::time_point now) const { return now >= retry_at_; }
    std::chrono::milliseconds Record(bool low_yield, Clock::time_point now) {
        delay_ = low_yield
                     ? std::min(delay_ * 2, std::chrono::milliseconds(250))
                     : std::chrono::milliseconds(10);
        retry_at_ = now + delay_;
        return delay_;
    }
    void Reset() {
        delay_ = std::chrono::milliseconds(10);
        retry_at_ = {};
    }

   private:
    std::chrono::milliseconds delay_{10};
    Clock::time_point retry_at_{};
};

inline constexpr size_t kMemoryEvictionScanLimit = 4096;
inline constexpr size_t kMemoryEvictionShardLimit = 64;

}  // namespace mooncake
