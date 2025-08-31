///// Otter: DeviceLogPolicy – centralizes device log flush cadence (every N frames or on error).
///// Schneefuchs: Avoids scattered modulo checks; ASCII-only; host-side policy wrapper; uses flushDeviceLogToHost(0).
///// Maus: Central policy; flush on cadence or immediate on CUDA error; strict host/device log separation.
///// Datei: src/device_log_policy.hpp

#pragma once
#include <cstdint>
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp" // provides flushDeviceLogToHost(int streamIdOrZero)

namespace otterdream {

class DeviceLogPolicy {
public:
    explicit DeviceLogPolicy(uint32_t cadence = 30) noexcept
        : cadence_(cadence ? cadence : 30) {}

    // Call each frame.
    void flushIfNeeded(uint64_t frameId, bool cudaHadError) const {
        if (cudaHadError || ((frameId % cadence_) == 0ull)) {
            flushDeviceLogToHost(0);
            if (cudaHadError) {
                LUCHS_LOG_HOST("[DevLog] immediate flush due to CUDA error; frame=%llu",
                               (unsigned long long)frameId);
            }
        }
    }

private:
    uint32_t cadence_;
};

} // namespace otterdream
