// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
#include "disk_watermark_eviction.h"

#include <gtest/gtest.h>

namespace mooncake::test {
using namespace std::chrono_literals;

TEST(DiskWatermarkEvictionTest, FinishesCycleBelowHighBeforeRearming) {
    DiskWatermarkEviction eviction;
    const auto now = DiskWatermarkEviction::Clock::time_point{};
    EXPECT_FALSE(eviction.Begin(90, 90, 80, now));
    EXPECT_FALSE(eviction.RetryDelay(now));
    ASSERT_TRUE(eviction.Begin(99, 90, 80, now));
    eviction.Finish(86, 80, true, false, now);
    EXPECT_EQ(eviction.RetryDelay(now), 0ms);
    ASSERT_TRUE(eviction.Begin(86, 90, 80, now));
    eviction.Finish(80, 80, true, false, now);
    EXPECT_FALSE(eviction.RetryDelay(now));
    EXPECT_FALSE(eviction.Begin(85, 90, 80, now));
    EXPECT_TRUE(eviction.Begin(91, 90, 80, now));
}

TEST(DiskWatermarkEvictionTest, DoesNotOverlapPassesOrLoseConcurrentWrites) {
    DiskWatermarkEviction eviction;
    const auto now = DiskWatermarkEviction::Clock::time_point{};
    ASSERT_TRUE(eviction.Begin(99, 90, 80, now));
    // Even a temporarily low usage from prepared removals cannot cancel an
    // in-progress pass. Writes/rollbacks during notification may raise it
    // again.
    EXPECT_FALSE(eviction.Begin(70, 90, 80, now));
    eviction.Finish(85, 80, true, false, now);
    EXPECT_EQ(eviction.RetryDelay(now), 0ms);
    EXPECT_TRUE(eviction.Begin(85, 90, 80, now));
}

TEST(DiskWatermarkEvictionTest, BlockedPassBacksOffDespiteIncomingWrites) {
    DiskWatermarkEviction eviction;
    auto now = DiskWatermarkEviction::Clock::time_point{};
    for (const auto delay : {100ms, 200ms, 400ms, 800ms, 1000ms, 1000ms}) {
        ASSERT_TRUE(eviction.Begin(99, 90, 80, now));
        eviction.Finish(95, 80, true, true, now);
        EXPECT_EQ(eviction.RetryDelay(now), delay);
        EXPECT_FALSE(eviction.Begin(100, 90, 80, now + delay - 1ms));
        now += delay;
    }
    // Recovery clears the backoff; continued progress can run next bucket.
    ASSERT_TRUE(eviction.Begin(99, 90, 80, now));
    eviction.Finish(85, 80, true, false, now);
    EXPECT_EQ(eviction.RetryDelay(now), 0ms);
    ASSERT_TRUE(eviction.Begin(85, 90, 80, now));
    eviction.Finish(85, 80, false, false, now);
    EXPECT_EQ(eviction.RetryDelay(now), 100ms);
}

TEST(DiskWatermarkEvictionTest, NoCandidatesBacksOffAndRemoveAllCancelsCycle) {
    DiskWatermarkEviction eviction;
    const auto now = DiskWatermarkEviction::Clock::time_point{};
    ASSERT_TRUE(eviction.Begin(99, 90, 80, now));
    eviction.Finish(99, 80, false, false, now);
    EXPECT_EQ(eviction.RetryDelay(now), 100ms);
    EXPECT_FALSE(eviction.Begin(0, 90, 80, now));
    EXPECT_FALSE(eviction.RetryDelay(now));
    EXPECT_TRUE(eviction.Begin(95, 90, 80, now));
}

TEST(DiskWatermarkEvictionTest, InterleavedWritesDoNotStopCleanupAtHigh) {
    DiskWatermarkEviction eviction;
    const auto now = DiskWatermarkEviction::Clock::time_point{};
    int64_t used = 99;
    int passes = 0;
    for (int i = 0; i < 10; ++i) {
        if (eviction.Begin(used, 90, 80, now)) {
            ++passes;
            used -= 6;  // One bounded cleanup pass.
            eviction.Finish(used, 80, true, false, now);
        }
        used += 1;  // One offload bucket, even while cleanup is active.
    }
    EXPECT_EQ(passes, 4);
    EXPECT_EQ(used, 85);
    EXPECT_FALSE(eviction.RetryDelay(now));
}

}  // namespace mooncake::test
