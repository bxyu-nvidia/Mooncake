// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>

namespace mooncake {

// Protected by the backend metadata lock. A bounded pass may finish below the
// high watermark but above the low watermark; keep that cycle active. Failed
// or blocked passes retain the cycle but must not retry on every incoming
// write.
class DiskWatermarkEviction {
   public:
    using Clock = std::chrono::steady_clock;

    bool Begin(int64_t used, int64_t high, int64_t low,
               Clock::time_point now = Clock::now()) {
        if (running_) return false;
        if (used <= low) {
            active_ = false;
            backoff_ = std::chrono::milliseconds(0);
            retry_at_ = {};
        } else if (used > high) {
            active_ = true;
        }
        if (!active_ || now < retry_at_) return false;
        running_ = true;
        return true;
    }

    void Finish(int64_t used, int64_t low, bool made_progress, bool blocked,
                Clock::time_point now = Clock::now()) {
        running_ = false;
        active_ = used > low;
        if (active_ && (blocked || !made_progress)) {
            backoff_ = std::clamp(backoff_ * 2, std::chrono::milliseconds(100),
                                  std::chrono::milliseconds(1000));
        } else {
            backoff_ = std::chrono::milliseconds(0);
        }
        retry_at_ = now + backoff_;
    }

    std::optional<std::chrono::milliseconds> RetryDelay(
        Clock::time_point now = Clock::now()) const {
        if (!active_) return std::nullopt;
        if (running_) return std::chrono::milliseconds(100);
        return std::max(
            std::chrono::milliseconds(0),
            std::chrono::ceil<std::chrono::milliseconds>(retry_at_ - now));
    }

   private:
    bool active_{false};
    bool running_{false};
    Clock::time_point retry_at_{};
    std::chrono::milliseconds backoff_{0};
};

}  // namespace mooncake
