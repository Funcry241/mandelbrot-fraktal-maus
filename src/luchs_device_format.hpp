///// Otter: Header-only device-safe tiny formatter; adds hex/pointer/bool helpers; zero-padded fixed floats.
///*** Schneefuchs: Keine __device__-CRT-Redeclarations; deterministisch, bounded writes; ASCII-only; host/device inline.
///*** Maus: Minimaler, stabiler Formatter fuer Device-Logs; kein snprintf; Basename/KV-Helpers; Header/Source synchron.
///*** Datei: src/luchs_device_format.hpp

#pragma once
#include <cstddef>
#include <cstdint>

namespace luchs {

// -----------------------------------------------------------------------------
// Internal: safe char put (appends one char and maintains 0-termination)
// Returns the new logical length n+1 (even if truncated).
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_putc(char* dst, int cap, int n, char c) {
    if (cap <= 0) return n;
    if (n < cap - 1) {
        dst[n++] = c;
        dst[n] = '\0';
        return n;
    }
    // already full; keep last char a null terminator
    dst[cap - 1] = '\0';
    return n;
}

// -----------------------------------------------------------------------------
// Append a null-terminated string
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_str(char* dst, int cap, int n, const char* s) {
    if (!s) return d_append_str(dst, cap, n, "(null)");
    for (char c = *s; c != '\0'; c = *++s) {
        if (n + 1 >= cap) { if (cap > 0) dst[cap - 1] = '\0'; return n; }
        dst[n++] = c;
    }
    if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0';
    return n;
}

// -----------------------------------------------------------------------------
// Append at most len characters from s (no need to be null-terminated).
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_strn(char* dst, int cap, int n, const char* s, int len) {
    if (!s) return d_append_str(dst, cap, n, "(null)");
    for (int i = 0; i < len; ++i) {
        char c = s[i];
        if (n + 1 >= cap) { if (cap > 0) dst[cap - 1] = '\0'; return n; }
        dst[n++] = c;
    }
    if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0';
    return n;
}

// -----------------------------------------------------------------------------
// Append a single character
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_char(char* dst, int cap, int n, char c) {
    return d_putc(dst, cap, n, c);
}

// -----------------------------------------------------------------------------
// Append unsigned decimal (u64)
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_u64(char* dst, int cap, int n, unsigned long long v) {
    // buffer for reversed digits
    char tmp[32];
    int  k = 0;
    if (v == 0) {
        return d_putc(dst, cap, n, '0');
    }
    while (v > 0 && k < (int)sizeof(tmp)) {
        unsigned digit = (unsigned)(v % 10ull);
        tmp[k++] = (char)('0' + digit);
        v /= 10ull;
    }
    while (k-- > 0) {
        n = d_putc(dst, cap, n, tmp[k]);
    }
    return n;
}

// -----------------------------------------------------------------------------
// Append signed decimal (i64)
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_i64(char* dst, int cap, int n, long long v) {
    if (v < 0) {
        n = d_putc(dst, cap, n, '-');
        unsigned long long uv = (unsigned long long)(-(v + 1)) + 1ull;
        return d_append_u64(dst, cap, n, uv);
    }
    return d_append_u64(dst, cap, n, (unsigned long long)v);
}

// -----------------------------------------------------------------------------
// Append fixed-point float: decimals >= 0
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_float_fixed(char* dst, int cap, int n, float x, int decimals) {
    if (decimals < 0) decimals = 0;
    if (x < 0.f) {
        n = d_putc(dst, cap, n, '-');
        x = -x;
    }
    // integer part
    unsigned long long ip = (unsigned long long)x;
    n = d_append_u64(dst, cap, n, ip);

    if (decimals == 0) return n;

    // fractional part
    n = d_putc(dst, cap, n, '.');
    float frac = x - (float)ip;
    for (int i = 0; i < decimals; ++i) {
        frac *= 10.f;
        unsigned d = (unsigned)frac;
        n = d_putc(dst, cap, n, (char)('0' + (d % 10)));
        frac -= (float)d;
    }
    return n;
}

// -----------------------------------------------------------------------------
// Append hex for u64 with 0x prefix
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_hex_u64(char* dst, int cap, int n, unsigned long long v) {
    n = d_append_str(dst, cap, n, "0x");
    char tmp[16];
    int  k = 0;
    if (v == 0) {
        return d_putc(dst, cap, n, '0');
    }
    while (v && k < 16) {
        unsigned nib = (unsigned)(v & 0xfull);
        tmp[k++] = (char)((nib < 10) ? ('0' + nib) : ('a' + (nib - 10)));
        v >>= 4ull;
    }
    while (k-- > 0) n = d_putc(dst, cap, n, tmp[k]);
    return n;
}

// -----------------------------------------------------------------------------
// Append pointer (platform-sized, shown via u64 cast)
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_ptr(char* dst, int cap, int n, const void* p) {
    unsigned long long v = (unsigned long long)(uintptr_t)p;
    return d_append_hex_u64(dst, cap, n, v);
}

// -----------------------------------------------------------------------------
// Bool helpers: "0" or "1"
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_bool01(char* dst, int cap, int n, bool v) {
    return d_putc(dst, cap, n, v ? '1' : '0');
}

// -----------------------------------------------------------------------------
// Terminate (ensure 0-termination within cap)
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
void d_terminate(char* dst, int cap, int n) {
    if (cap <= 0) return;
    dst[(n < cap ? n : cap - 1)] = '\0';
}

} // namespace luchs
