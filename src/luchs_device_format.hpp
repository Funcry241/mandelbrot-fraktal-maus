// MAUS: device-safe tiny formatter (bounded; no CRT redeclare; ASCII only)
// Datei: src/luchs_device_format.hpp
// 🐭 Maus: Header-only helpers to build log strings on device without snprintf.
// 🦦 Otter: Supports strings, ints, and fixed-point floats with chosen decimals.
// 🦊 Schneefuchs: No __device__ redeclarations of CRT; deterministic, bounded writes.

#pragma once
#include <cstddef>
#include <cstdint>

namespace luchs {

// ---- Internal: safe char put ------------------------------------------------
__host__ __device__ __forceinline__
int d_put(char* dst, int cap, int n, char c) {
    if (n + 1 < cap) { dst[n] = c; dst[n + 1] = '\0'; }
    return n + 1;
}

// ---- Append C-string (null-terminated) --------------------------------------
__host__ __device__ __forceinline__
int d_append_str(char* dst, int cap, int n, const char* s) {
    if (!s) return d_append_str(dst, cap, n, "(null)");
    for (char c = *s; c != '\0'; c = *++s) {
        if (n + 1 >= cap) break;
        dst[n++] = c;
    }
    if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0';
    return n;
}

// ---- Append single char -----------------------------------------------------
__host__ __device__ __forceinline__
int d_append_char(char* dst, int cap, int n, char c) {
    return d_put(dst, cap, n, c);
}

// ---- Append unsigned integer (base10) ---------------------------------------
__host__ __device__ __forceinline__
int d_append_u64(char* dst, int cap, int n, unsigned long long v) {
    char buf[32];
    int  k = 0;
    do {
        unsigned digit = static_cast<unsigned>(v % 10ULL);
        buf[k++] = static_cast<char>('0' + digit);
        v /= 10ULL;
    } while (v && k < (int)sizeof(buf));
    // reverse
    while (k--) {
        if (n + 1 >= cap) { if (cap > 0) dst[cap - 1] = '\0'; return n; }
        dst[n++] = buf[k];
    }
    if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0';
    return n;
}

__host__ __device__ __forceinline__
int d_append_i64(char* dst, int cap, int n, long long v) {
    if (v < 0) {
        n = d_put(dst, cap, n, '-');
        // careful with LLONG_MIN
        unsigned long long uv = static_cast<unsigned long long>(-(v + 1)) + 1ULL;
        return d_append_u64(dst, cap, n, uv);
    } else {
        return d_append_u64(dst, cap, n, static_cast<unsigned long long>(v));
    }
}

__host__ __device__ __forceinline__
int d_append_int(char* dst, int cap, int n, int v) {
    return d_append_i64(dst, cap, n, static_cast<long long>(v));
}

// ---- Append fixed-point float (rounded) -------------------------------------
__host__ __device__ __forceinline__
int d_append_float_fixed(char* dst, int cap, int n, float fv, int decimals) {
    if (decimals < 0) decimals = 0;
    if (decimals > 9) decimals = 9; // keep scale in 64-bit safe range

    // sign
    bool neg = (fv < 0.0f);
    float av = neg ? -fv : fv;

    // scale = 10^decimals
    unsigned long long scale = 1ULL;
    for (int i = 0; i < decimals; ++i) scale *= 10ULL;

    // rounded fixed-point integer representation
    // Use double for intermediate to reduce rounding error in conversion
    double scaled_d = static_cast<double>(av) * static_cast<double>(scale) + 0.5;
    unsigned long long fixed = static_cast<unsigned long long>(scaled_d);

    unsigned long long intPart = fixed / scale;
    unsigned long long fracPart = fixed - intPart * scale;

    if (neg) n = d_put(dst, cap, n, '-');
    n = d_append_u64(dst, cap, n, intPart);

    if (decimals > 0) {
        n = d_put(dst, cap, n, '.');
        // zero-pad fractional part
        unsigned long long pad = scale / 10ULL;
        for (int i = 1; i < decimals; ++i) {
            if (fracPart < pad) {
                n = d_put(dst, cap, n, '0');
            }
            pad /= 10ULL;
            if (pad == 0) break;
        }
        n = d_append_u64(dst, cap, n, fracPart);
    }

    if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0';
    return n;
}

// ---- Terminate explicitly (safety) ------------------------------------------
__host__ __device__ __forceinline__
void d_terminate(char* dst, int cap, int n) {
    if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0';
}

} // namespace luchs
