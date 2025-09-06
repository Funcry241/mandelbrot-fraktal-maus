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
int d_put(char* dst, int cap, int n, char c) {
    if (cap > 0 && n >= 0) {
        if (n + 1 < cap) {
            dst[n] = c;
            dst[n + 1] = '\0';
        } else {
            // ensure last byte stays terminated
            dst[cap - 1] = '\0';
        }
    }
    return n + 1;
}

// -----------------------------------------------------------------------------
// Append C-string (null-terminated). Safe if s == nullptr.
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
    if (!s || len <= 0) { if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0'; return n; }
    for (int i = 0; i < len; ++i) {
        if (n + 1 >= cap) { if (cap > 0) dst[cap - 1] = '\0'; return n; }
        dst[n++] = s[i];
    }
    if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0';
    return n;
}

// -----------------------------------------------------------------------------
// Append single char
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_char(char* dst, int cap, int n, char c) {
    return d_put(dst, cap, n, c);
}

// -----------------------------------------------------------------------------
// Append unsigned/signed integer (base10)
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_u64(char* dst, int cap, int n, unsigned long long v) {
    char buf[32];
    int  k = 0;
    do {
        unsigned digit = static_cast<unsigned>(v % 10ULL);
        buf[k++] = static_cast<char>('0' + digit);
        v /= 10ULL;
    } while (v && k < (int)sizeof(buf));
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
        // Careful with LLONG_MIN
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

// -----------------------------------------------------------------------------
// Append hex (lowercase). Raw (no 0x) and prefixed (0x...).
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_hex_u64_raw(char* dst, int cap, int n, unsigned long long v) {
    char buf[16]; // 64-bit → max 16 hex digits
    int  k = 0;
    do {
        unsigned nib = static_cast<unsigned>(v & 0xFULL);
        buf[k++] = (nib < 10) ? static_cast<char>('0' + nib)
                              : static_cast<char>('a' + (nib - 10));
        v >>= 4;
    } while (v && k < (int)sizeof(buf));
    while (k--) {
        if (n + 1 >= cap) { if (cap > 0) dst[cap - 1] = '\0'; return n; }
        dst[n++] = buf[k];
    }
    if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0';
    return n;
}

__host__ __device__ __forceinline__
int d_append_hex_u64(char* dst, int cap, int n, unsigned long long v) {
    n = d_append_str(dst, cap, n, "0x");
    return d_append_hex_u64_raw(dst, cap, n, v);
}

// Pointer as 0x...
__host__ __device__ __forceinline__
int d_append_ptr(char* dst, int cap, int n, const void* p) {
    unsigned long long addr = (unsigned long long)( (size_t)p );
    return d_append_hex_u64(dst, cap, n, addr);
}

// -----------------------------------------------------------------------------
// Append bool ("true"/"false") and compact bool ('0'/'1')
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_bool(char* dst, int cap, int n, bool v) {
    return v ? d_append_str(dst, cap, n, "true")
             : d_append_str(dst, cap, n, "false");
}

__host__ __device__ __forceinline__
int d_append_bool01(char* dst, int cap, int n, bool v) {
    return d_put(dst, cap, n, v ? '1' : '0');
}

// -----------------------------------------------------------------------------
// Append fixed-point float (rounded, decimals in [0,9]).
// Writes exactly 'decimals' fractional digits with correct zero padding.
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_float_fixed(char* dst, int cap, int n, float fv, int decimals) {
    if (decimals < 0) decimals = 0;
    if (decimals > 9) decimals = 9; // keep scale in 64-bit safe range

    // Handle special cases conservatively (no IEEE strings on device)
    // NaN check: (x != x)
    if (!(fv == fv)) return d_append_str(dst, cap, n, "nan");
    // +/- inf heuristic: enormous values
    const float INF_T = 3.4e38f;
    if (fv >  INF_T) return d_append_str(dst, cap, n, "inf");
    if (fv < -INF_T) return d_append_str(dst, cap, n, "-inf");

    bool neg = (fv < 0.0f);
    float av = neg ? -fv : fv;

    // scale = 10^decimals
    unsigned long long scale = 1ULL;
    for (int i = 0; i < decimals; ++i) scale *= 10ULL;

    // rounded fixed-point integer representation
    // use double for intermediate to reduce rounding error
    double scaled_d = static_cast<double>(av) * static_cast<double>(scale) + 0.5;
    unsigned long long fixed = (unsigned long long)(scaled_d); // trunc == floor since positive

    unsigned long long intPart = (decimals > 0) ? (fixed / scale) : fixed;
    unsigned long long fracPart = (decimals > 0) ? (fixed - intPart * scale) : 0ULL;

    if (neg) n = d_put(dst, cap, n, '-');
    n = d_append_u64(dst, cap, n, intPart);

    if (decimals > 0) {
        n = d_put(dst, cap, n, '.');

        // Write exactly 'decimals' digits with left zero padding.
        char fbuf[20];
        int  fk = 0;
        unsigned long long tmp = fracPart;
        do {
            unsigned d = (unsigned)(tmp % 10ULL);
            fbuf[fk++] = (char)('0' + d);
            tmp /= 10ULL;
        } while (tmp && fk < (int)sizeof(fbuf));

        while (fk < decimals && fk < (int)sizeof(fbuf)) {
            fbuf[fk++] = '0';
        }

        int toWrite = (fk < decimals) ? fk : decimals;
        for (int i = toWrite - 1; i >= 0; --i) {
            if (n + 1 >= cap) { if (cap > 0) dst[cap - 1] = '\0'; return n; }
            dst[n++] = fbuf[i];
        }
        if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0';
    }

    return n;
}

// -----------------------------------------------------------------------------
// Basename helper: append only the last path component of 'path'.
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_basename(char* dst, int cap, int n, const char* path) {
    if (!path) return d_append_str(dst, cap, n, "(null)");
    const char* base = path;
    for (const char* p = path; *p; ++p) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }
    return d_append_str(dst, cap, n, base);
}

// -----------------------------------------------------------------------------
// KV helpers: write "key=value" (no quotes added)
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
int d_append_kv_str(char* dst, int cap, int n, const char* key, const char* val) {
    n = d_append_str(dst, cap, n, key);
    n = d_put(dst, cap, n, '=');
    return d_append_str(dst, cap, n, val);
}

__host__ __device__ __forceinline__
int d_append_kv_int(char* dst, int cap, int n, const char* key, long long val) {
    n = d_append_str(dst, cap, n, key);
    n = d_put(dst, cap, n, '=');
    return d_append_i64(dst, cap, n, val);
}

__host__ __device__ __forceinline__
int d_append_kv_u64(char* dst, int cap, int n, const char* key, unsigned long long val) {
    n = d_append_str(dst, cap, n, key);
    n = d_put(dst, cap, n, '=');
    return d_append_u64(dst, cap, n, val);
}

__host__ __device__ __forceinline__
int d_append_kv_fixed(char* dst, int cap, int n, const char* key, float val, int decimals) {
    n = d_append_str(dst, cap, n, key);
    n = d_put(dst, cap, n, '=');
    return d_append_float_fixed(dst, cap, n, val, decimals);
}

// -----------------------------------------------------------------------------
// Terminate explicitly (safety)
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
void d_terminate(char* dst, int cap, int n) {
    if (cap > 0) dst[(n < cap ? n : cap - 1)] = '\0';
}

} // namespace luchs
