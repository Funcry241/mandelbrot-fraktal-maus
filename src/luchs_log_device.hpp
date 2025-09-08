///// Otter: Device-Log-Brücke – __FILE__/__LINE__ automatisch; Builder ohne snprintf; nutzt tiny Formatter.
///*** Schneefuchs: Header-only; keine CRT-Re-Decls im __device__; deterministisch; bounded writes; ASCII-only.
///*** Maus: Minimal, stabil; ein Makro (LUCHS_LOG_DEVICE) + ein Builder (LUCHS_LOG_DEVICE_BUILD); API bleibt schlank.
///*** Datei: src/luchs_log_device.hpp

#pragma once
#include <cuda_runtime.h>
#include "luchs_cuda_log_buffer.hpp"  // LOG_MESSAGE_MAX + LuchsLogger::deviceLog(...)
#include "luchs_device_format.hpp"    // d_append_* helpers

namespace luchs {

// Tiny device-side builder for log lines.
// Usage (device):
//   LUCHS_LOG_DEVICE_BUILD( __dl.s("it=").i(it).s(" norm=").f(norm,3) );
struct DevLog {
    char buf[LOG_MESSAGE_MAX];
    int  n;

    __device__ DevLog() : buf{0}, n(0) {}

    __device__ DevLog& s(const char* str) {
        n = d_append_str(buf, (int)sizeof(buf), n, str);
        return *this;
    }
    __device__ DevLog& c(char ch) {
        n = d_append_char(buf, (int)sizeof(buf), n, ch);
        return *this;
    }
    __device__ DevLog& i(long long v) {
        n = d_append_i64(buf, (int)sizeof(buf), n, v);
        return *this;
    }
    __device__ DevLog& u(unsigned long long v) {
        n = d_append_u64(buf, (int)sizeof(buf), n, v);
        return *this;
    }
    __device__ DevLog& f(float v, int decimals) {
        n = d_append_float_fixed(buf, (int)sizeof(buf), n, v, decimals);
        return *this;
    }
    __device__ DevLog& hex(unsigned long long v) {
        n = d_append_hex_u64(buf, (int)sizeof(buf), n, v);
        return *this;
    }
    __device__ DevLog& b(bool v) {
        n = d_append_bool01(buf, (int)sizeof(buf), n, v);
        return *this;
    }

    __device__ void send(const char* file, int line) {
        d_terminate(buf, (int)sizeof(buf), n);
        LuchsLogger::deviceLog(file, line, buf);
    }
};

} // namespace luchs

// -----------------------------------------------------------------------------
// Public device logging macros
// -----------------------------------------------------------------------------
#define LUCHS_LOG_DEVICE_BUILD(expr) do { ::luchs::DevLog __dl; expr; __dl.send(__FILE__, __LINE__); } while(0)
#define LUCHS_LOG_DEVICE(msg)        do { LuchsLogger::deviceLog(__FILE__, __LINE__, (msg)); } while(0)
