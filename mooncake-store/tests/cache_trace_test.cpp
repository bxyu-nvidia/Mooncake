// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#include "replica.h"

#include <gtest/gtest.h>
#include <sstream>
#include <ylt/struct_json/json_reader.h>

namespace mooncake {
namespace {

std::vector<CacheReplicaTrace> ReadTrace(const std::string& log) {
    std::vector<CacheReplicaTrace> records;
    std::istringstream lines(log);
    std::string line;
    const std::string marker = "MAMBA_CACHE_TRACE ";
    while (std::getline(lines, line)) {
        auto pos = line.find(marker);
        if (pos == std::string::npos) continue;
        CacheReplicaTrace record;
        struct_json::from_json(record, line.substr(pos + marker.size()));
        records.push_back(std::move(record));
    }
    return records;
}

class CacheTraceTest : public ::testing::Test {
   protected:
    void SetUp() override {
        setenv("MOONCAKE_CACHE_TRACE", "1", 1);
        FLAGS_logtostderr = true;
        FLAGS_minloglevel = 0;
        testing::internal::CaptureStderr();
    }

    std::vector<CacheReplicaTrace> FinishCapture() {
        return ReadTrace(testing::internal::GetCapturedStderr());
    }
};

TEST_F(CacheTraceTest, MoveAndRetireDoNotReclaimDiskStorage) {
    const std::string key = "model\"quoted@group:0@" + std::string(64, 'a');
    {
        Replica original("/tmp/state", 1024, ReplicaStatus::PROCESSING);
        original.AttachCacheTrace(key, "tenant");
        original.mark_complete();
        Replica moved(std::move(original));
        moved.TraceRetired("evicted");
        moved.TraceRetired("evicted_again");
    }
    auto records = FinishCapture();
    ASSERT_EQ(records.size(), 4);
    EXPECT_EQ(records[0].event, "mooncake_replica_attached");
    EXPECT_EQ(records[1].event, "mooncake_replica_committed");
    EXPECT_EQ(records[2].event, "mooncake_replica_retired");
    EXPECT_EQ(records[3].event, "mooncake_replica_released");
    EXPECT_EQ(records[3].object_key, key);
    EXPECT_EQ(records[3].replica_id, records[0].replica_id);
    EXPECT_FALSE(records[3].storage_reclaimed_ns.has_value());
    ASSERT_TRUE(records[3].committed_ns.has_value());
    EXPECT_LE(*records[3].committed_ns, *records[3].retired_ns);
}

TEST_F(CacheTraceTest,
       MemoryReleaseReturnsAllocationBeforeReportingReclamation) {
    auto allocator = std::make_shared<OffsetBufferAllocator>(
        "trace-test", 0x100000000ULL, 16 * 1024 * 1024, "trace-test");
    {
        Replica original(allocator->allocate(1024), ReplicaStatus::COMPLETE);
        original.AttachCacheTrace("model@group:0@abcd", "default");
        Replica target("/tmp/unused", 1024, ReplicaStatus::COMPLETE);
        target = std::move(original);
        target.TraceRetired("detached_from_object");
        EXPECT_GT(allocator->size(), 0);
    }
    EXPECT_EQ(allocator->size(), 0);
    auto records = FinishCapture();
    ASSERT_EQ(records.size(), 3);
    const auto& released = records.back();
    EXPECT_EQ(released.event, "mooncake_replica_released");
    EXPECT_EQ(released.tier, "memory");
    ASSERT_TRUE(released.storage_reclaimed_ns.has_value());
    EXPECT_LE(*released.retired_ns, *released.storage_reclaimed_ns);
}

}  // namespace
}  // namespace mooncake
