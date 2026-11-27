// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
#include "memory_eviction.h"

#include <gtest/gtest.h>

#include <memory>
#include <unordered_set>
#include <vector>

namespace mooncake::test {
namespace {
struct Object {
    explicit Object(MemoryEvictionIndex<Object>& index, size_t id)
        : id(id), membership(&index, this) {}
    size_t id;
    MemoryEvictionIndex<Object>::Membership membership;
};
}  // namespace

TEST(MemoryEvictionIndexTest, MembershipTracksResidentLifetime) {
    MemoryEvictionIndex<Object> index;
    {
        Object first(index, 1);
        Object second(index, 2);
        EXPECT_EQ(index.size(), 0);
        first.membership.SetResident(true);
        first.membership.SetResident(true);
        second.membership.SetResident(true);
        EXPECT_EQ(index.size(), 2);
        first.membership.SetResident(false);
        EXPECT_EQ(index.size(), 1);
        first.membership.SetResident(true);  // e.g. promotion from disk
        EXPECT_EQ(index.size(), 2);
        std::vector<size_t> seen;
        EXPECT_EQ(index.Visit(10, [&](Object& o) { seen.push_back(o.id); }), 2);
        EXPECT_EQ(seen, (std::vector<size_t>{2, 1}));
    }
    EXPECT_EQ(index.size(), 0);
}

TEST(MemoryEvictionIndexTest, BoundedSamplesReachPastPinnedEntries) {
    MemoryEvictionIndex<Object> index;
    std::vector<std::unique_ptr<Object>> objects;
    for (size_t i = 0; i < 3 * kMemoryEvictionScanLimit; ++i) {
        auto object = std::make_unique<Object>(index, i);
        object->membership.SetResident(true);
        objects.push_back(std::move(object));
    }
    std::unordered_set<size_t> seen;
    for (size_t i = 0; i < 3; ++i) {
        const size_t count =
            index.Visit(kMemoryEvictionScanLimit, [&](Object& o) {
                EXPECT_TRUE(seen.insert(o.id).second);
                // Protected entries stay resident, but cannot starve later
                // ones.
            });
        EXPECT_EQ(count, kMemoryEvictionScanLimit);
    }
    EXPECT_EQ(seen.size(), objects.size());
}

TEST(MemoryEvictionIndexTest, DiskPopulationDoesNotIncreaseScanWork) {
    MemoryEvictionIndex<Object> index;
    std::vector<std::unique_ptr<Object>> objects;
    for (size_t i = 0; i < 100000; ++i) {
        auto object = std::make_unique<Object>(index, i);
        object->membership.SetResident(i < 3);
        objects.push_back(std::move(object));
    }
    size_t visited = 0;
    EXPECT_EQ(
        index.Visit(kMemoryEvictionScanLimit, [&](Object&) { ++visited; }), 3);
    EXPECT_EQ(visited, 3);
    objects[0]->membership.SetResident(false);
    objects[1].reset();
    EXPECT_EQ(index.size(), 1);
    EXPECT_EQ(index.Visit(kMemoryEvictionScanLimit,
                          [](Object& o) { EXPECT_EQ(o.id, 2); }),
              1);
}

TEST(MemoryEvictionBackoffTest, RepeatedPressureCannotBypassCooldown) {
    MemoryEvictionBackoff backoff;
    auto now = MemoryEvictionBackoff::Clock::now();
    EXPECT_TRUE(backoff.Ready(now));
    for (int expected : {20, 40, 80, 160, 250, 250}) {
        const auto delay = backoff.Record(true, now);
        EXPECT_EQ(delay.count(), expected);
        EXPECT_FALSE(backoff.Ready(now));
        EXPECT_FALSE(backoff.Ready(now + delay - std::chrono::milliseconds(1)));
        now += delay;
        EXPECT_TRUE(backoff.Ready(now));
    }
    EXPECT_EQ(backoff.Record(false, now).count(), 10);
    backoff.Reset();
    EXPECT_TRUE(backoff.Ready(now));
    EXPECT_EQ(backoff.Record(true, now).count(), 20);
}
}  // namespace mooncake::test
