// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#include "mamba_direct_use.h"

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

namespace mooncake::test {

TEST(MambaDirectUsePolicyTest, RetryDoesNotProtectOrDeleteReinsertedObject) {
    MambaDirectUsePolicy policy;
    int removals = 0;
    auto remove = [&](const auto&, const auto&) {
        ++removals;
        return true;
    };
    EXPECT_EQ(policy.Report("tenant/key", "use-a", "key", "tenant"), 1);
    policy.Drain(remove);
    EXPECT_EQ(removals, 1);
    // A retransmission must neither count a second use nor delete a new
    // residency.
    EXPECT_EQ(policy.Report("tenant/key", "use-a", "key", "tenant"), 1);
    policy.Drain(remove);
    EXPECT_EQ(removals, 1);
    EXPECT_EQ(policy.Report("tenant/key", "use-b", "key", "tenant"), 2);
    EXPECT_EQ(policy.Report("tenant/key", "use-c", "key", "tenant"), 2);
    policy.Drain(remove);
    EXPECT_EQ(removals, 1);
}

TEST(MambaDirectUsePolicyTest, SecondUseCancelsBlockedRemoval) {
    MambaDirectUsePolicy policy;
    int attempts = 0;
    policy.Report("tenant/key", "use-a", "key", "tenant");
    policy.Drain([&](const auto& key, const auto& tenant) {
        EXPECT_EQ(key, "key");
        EXPECT_EQ(tenant, "tenant");
        ++attempts;
        return false;  // Active lease or write.
    });
    EXPECT_EQ(policy.Report("tenant/key", "use-b", "key", "tenant"), 2);
    policy.Drain([&](const auto&, const auto&) {
        ++attempts;
        return true;
    });
    EXPECT_EQ(attempts, 1);
}

TEST(MambaDirectUsePolicyTest, BlockedRemovalRetriesWithoutNewUses) {
    MambaDirectUsePolicy policy;
    policy.Report("tenant/key", "use-a", "key", "tenant");
    int attempts = 0;
    auto remove = [&](const auto&, const auto&) { return ++attempts == 2; };
    policy.Drain(remove);
    policy.Drain(remove);
    policy.Drain(remove);
    EXPECT_EQ(attempts, 2);
}

TEST(MambaDirectUsePolicyTest, TenantsAndGroupsHaveIndependentProtection) {
    MambaDirectUsePolicy policy;
    policy.Report("tenant-a/group0", "use-a", "group0", "tenant-a");
    policy.Report("tenant-a/group0", "use-b", "group0", "tenant-a");
    policy.Report("tenant-b/group0", "use-a", "group0", "tenant-b");
    policy.Report("tenant-a/group1", "use-a", "group1", "tenant-a");
    std::vector<std::string> removed;
    policy.Drain([&](const auto& key, const auto& tenant) {
        removed.push_back(tenant + "/" + key);
        return true;
    });
    EXPECT_EQ(removed,
              (std::vector<std::string>{"tenant-b/group0", "tenant-a/group1"}));
}

TEST(MambaDirectUsePolicyTest, BoundedDrainMakesProgressPastBusyObjects) {
    MambaDirectUsePolicy policy;
    policy.Report("busy", "use-a", "busy", "tenant");
    policy.Report("ready", "use-a", "ready", "tenant");
    std::vector<std::string> attempts;
    auto remove = [&](const auto& key, const auto&) {
        attempts.push_back(key);
        return key == "ready";
    };
    policy.Drain(remove, 1);
    policy.Drain(remove, 1);
    EXPECT_EQ(attempts, (std::vector<std::string>{"busy", "ready"}));
}

TEST(MambaDirectUsePolicyTest, ConcurrentDuplicateReportsCountOnlyOneUse) {
    MambaDirectUsePolicy policy;
    std::vector<std::thread> threads;
    std::atomic<int> unprotected{0};
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&] {
            if (policy.Report("key", "same-use", "key", "tenant") == 1)
                ++unprotected;
        });
    }
    for (auto& thread : threads) thread.join();
    EXPECT_EQ(unprotected.load(), 8);
    EXPECT_EQ(policy.Report("key", "another-use", "key", "tenant"), 2);
    int removals = 0;
    policy.Drain([&](const auto&, const auto&) {
        ++removals;
        return true;
    });
    EXPECT_EQ(removals, 0);
}

}  // namespace mooncake::test
