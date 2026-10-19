// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#include "cache_trace.h"

#include <gtest/gtest.h>
#include <thread>
#include <ylt/struct_json/json_reader.h>

namespace mooncake {
namespace {

TEST(CacheTraceTest, OnlyRemovalEmitsAccumulatedCounts) {
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = 0;
    testing::internal::CaptureStderr();
    CacheObjectTrace trace;
    std::thread reader([&] {
        for (int i = 0; i < 1000; ++i) {
            trace.lookup_hits.fetch_add(1, std::memory_order_relaxed);
            trace.read_requests.fetch_add(1, std::memory_order_relaxed);
        }
    });
    for (int i = 0; i < 1000; ++i)
        trace.lookup_hits.fetch_add(1, std::memory_order_relaxed);
    reader.join();
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "");

    testing::internal::CaptureStderr();
    const std::string key = "model\"quoted@group:0@abcd";
    trace.Removed(key, "tenant");
    trace.Removed(key, "tenant");
    const auto log = testing::internal::GetCapturedStderr();
    const std::string marker = "MAMBA_CACHE_TRACE ";
    const auto start = log.find(marker);
    ASSERT_NE(start, std::string::npos);
    EXPECT_EQ(log.find(marker, start + marker.size()), std::string::npos);
    CacheRemovalRecord record;
    struct_json::from_json(record, log.substr(start + marker.size()));
    EXPECT_EQ(record.event, "hash_removed");
    EXPECT_EQ(record.object_key, key);
    EXPECT_EQ(record.prefix_hash, "abcd");
    EXPECT_EQ(record.tenant_id, "tenant");
    EXPECT_EQ(record.lookup_hit_count, 2000);
    EXPECT_EQ(record.read_request_count, 1000);
    EXPECT_GE(record.removed_ns, record.created_ns);
    EXPECT_GE(record.lifetime_ns, 0);
}

}  // namespace
}  // namespace mooncake
