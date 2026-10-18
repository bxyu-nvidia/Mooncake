// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#include "cache_inventory.h"

#include <gtest/gtest.h>

namespace mooncake {
namespace {
const std::vector<CacheInventoryNamespace> layout{
    {"model", "m0", "mamba"},
    {"model", "m1", "mamba"},
    {"model", "attn", "attention"},
    {"model", "lineage", "mamba_lineage"},
    {"model", "retired", "mamba_retired"}};

TEST(CacheInventoryTest, ClassifiesPayloadsIndependentOfMarkerScanOrder) {
    CacheInventoryAccumulator inventory(layout);
    inventory.Add("lineage@tip", 1, true);
    inventory.Add("retired@old", 1, true);
    for (const auto* hash : {"root", "tip", "old"}) {
        inventory.Add(std::string("m0@") + hash, 100, true);
        inventory.Add(std::string("m1@") + hash, 200, true);
        inventory.Add(std::string("attn@") + hash, 50, true);
    }
    inventory.Add("lineage@old", 1, true);
    const auto counts = inventory.Finish().counts;
    for (const auto* category : {"anchor_or_unknown", "tip", "retired"}) {
        const auto prefix = std::string("mamba.") + category + ".";
        EXPECT_EQ(counts.at(prefix + "complete_prefixes"), 1U);
        EXPECT_EQ(counts.at(prefix + "incomplete_prefixes"), 0U);
        EXPECT_EQ(counts.at(prefix + "ready_objects"), 2U);
        EXPECT_EQ(counts.at(prefix + "ready_bytes"), 300U);
    }
    EXPECT_EQ(counts.at("mamba.complete_prefixes"), 3U);
    EXPECT_EQ(counts.at("mamba.ready_bytes"), 900U);
    EXPECT_EQ(counts.at("attention.ready_bytes"), 150U);
}

TEST(CacheInventoryTest, CountsPartialRetiredPayloadsAndPendingBytes) {
    CacheInventoryAccumulator inventory(layout);
    inventory.Add("m0@old", 100, true);
    inventory.Add("m1@old", 200, false);
    inventory.Add("retired@old", 1, true);  // Lineage may already be evicted.
    const auto counts = inventory.Finish().counts;
    EXPECT_EQ(counts.at("mamba.retired.complete_prefixes"), 0U);
    EXPECT_EQ(counts.at("mamba.retired.incomplete_prefixes"), 1U);
    EXPECT_EQ(counts.at("mamba.retired.ready_objects"), 1U);
    EXPECT_EQ(counts.at("mamba.retired.ready_bytes"), 100U);
    EXPECT_EQ(counts.at("mamba.retired.pending_objects"), 1U);
    EXPECT_EQ(counts.at("mamba.retired.pending_bytes"), 200U);
    EXPECT_EQ(counts.at("mamba.incomplete_prefixes"), 1U);
}

TEST(CacheInventoryTest, OrphanAndUnfinishedMarkersAreNotCheckpoints) {
    CacheInventoryAccumulator inventory(layout);
    inventory.Add("lineage@gone", 1, true);
    inventory.Add("retired@gone", 1, true);
    inventory.Add("m0@present", 100, true);
    inventory.Add("m1@present", 100, true);
    inventory.Add("lineage@present", 1, false);
    inventory.Add("retired@present", 1, false);
    const auto counts = inventory.Finish().counts;
    EXPECT_EQ(counts.at("mamba.complete_prefixes"), 1U);
    EXPECT_EQ(counts.at("mamba.incomplete_prefixes"), 0U);
    EXPECT_EQ(counts.at("mamba.anchor_or_unknown.complete_prefixes"), 1U);
    EXPECT_EQ(counts.at("mamba.tip.complete_prefixes"), 0U);
    EXPECT_EQ(counts.at("mamba.retired.complete_prefixes"), 0U);
    EXPECT_EQ(counts.at("mamba.pending_objects"), 0U);
}

TEST(CacheInventoryTest, SameHashInOtherScopeIsNotRetired) {
    auto namespaces = layout;
    namespaces.push_back({"other", "other-mamba", "mamba"});
    CacheInventoryAccumulator inventory(namespaces);
    inventory.Add("m0@hash", 100, true);
    inventory.Add("m1@hash", 100, true);
    inventory.Add("other-mamba@hash", 400, true);
    inventory.Add("retired@hash", 1, true);
    const auto counts = inventory.Finish().counts;
    EXPECT_EQ(counts.at("mamba.retired.ready_bytes"), 200U);
    EXPECT_EQ(counts.at("mamba.anchor_or_unknown.ready_bytes"), 400U);
    EXPECT_EQ(counts.at("mamba.complete_prefixes"), 2U);
}

TEST(CacheInventoryTest, MissingShardAndLostMarkerRemainVisible) {
    CacheInventoryAccumulator inventory(layout);
    inventory.Add("m0@hash", 100, true);
    // A missing marker does not prove that the payload is a genuine anchor.
    const auto counts = inventory.Finish().counts;
    EXPECT_EQ(counts.at("mamba.anchor_or_unknown.incomplete_prefixes"), 1U);
    EXPECT_EQ(counts.at("mamba.anchor_or_unknown.ready_bytes"), 100U);
    EXPECT_EQ(counts.at("mamba.complete_prefixes"), 0U);
}

TEST(CacheInventoryTest, OldLayoutsStillWorkAndEmptyCategoriesExportZero) {
    CacheInventoryAccumulator inventory({{"model", "m0", "mamba"}});
    inventory.Add("m0@hash", 100, true);
    inventory.Add("retired@hash", 1, true);  // Namespace was not advertised.
    const auto counts = inventory.Finish().counts;
    EXPECT_EQ(counts.at("mamba.anchor_or_unknown.complete_prefixes"), 1U);
    EXPECT_EQ(counts.at("mamba.tip.ready_bytes"), 0U);
    EXPECT_EQ(counts.at("mamba.retired.ready_bytes"), 0U);
}

TEST(CacheInventoryTest, MarkerNamespaceRequiresMatchingPayloadScope) {
    const std::vector<CacheInventoryNamespace> invalid{
        {"model", "m0", "mamba"}, {"other", "retired", "mamba_retired"}};
    EXPECT_THROW(CacheInventoryAccumulator{invalid}, std::invalid_argument);
}
}  // namespace
}  // namespace mooncake
