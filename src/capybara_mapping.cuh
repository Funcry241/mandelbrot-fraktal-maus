///// Otter: Pixel->Complex mapping hygiene for Capybara (Hi/Lo + exact step split).
///// Schneefuchs: Header-only; safe include anywhere; ASCII-only; rate-limited device logs via capy_* helpers.
///// Maus: Returns both classic double2 and CapyHiLo2; no API break for existing kernels.
///// Datei: src/capybara_mapping.cuh

#pragma once
#include <math.h>
#include <stdint.h>
#include <vector_types.h>    // double2
#include <cmath>             // std::fma on host

#include "capybara_math.cuh" // CapyHiLo / CapyHiLo2 / capy_map_center_step(...)

#if defined(__CUDACC__)
  #define CAPY_HD __host__ __device__ __forceinline__
  #define CAPY_D  __device__ __forceinline__
#else
  #define CAPY_HD inline
  #define CAPY_D  inline
#endif

// Default log stride for CAPY debugging (can be overridden via build flags).
// CAPY_LOG_RATE = 1  -> prüfen jeden GID
// CAPY_LOG_RATE = N  -> nur 1 von N GIDs wird geloggt (via capy_should_log).
#ifndef CAPY_LOG_RATE
#define CAPY_LOG_RATE 256u
#endif

// Small wrapper to get FMA on host/device uniformly.
CAPY_HD double capy_fma(double a, double b, double c) noexcept {
#if defined(__CUDA_ARCH__)
    return fma(a,b,c);
#else
    return std::fma(a,b,c);
#endif
}

// -----------------------------------------------------------------------------
// Schrittweiten aus pixelScale + zoom ableiten (einheitliche Semantik).
// Korrektur: keine max()-Isotropisierung hier; jede Achse nutzt ihre Skala.
// Y negativ (GL-Raster nach unten). PixelScale ist **zoomfrei**.
// -----------------------------------------------------------------------------
CAPY_HD void capy_pixel_steps_from_zoom_scale(double sx, double sy,
                                              int width, double zoom,
                                              double& stepX, double& stepY) noexcept
{
    const double invZoom = (zoom != 0.0) ? (1.0 / zoom) : 1.0;

    // Normalpfad: direkte Achsen-Skalen, Zoom erst hier anwenden.
    if (!(sx == 0.0 && sy == 0.0)) {
        stepX = sx * invZoom;
        stepY = -sy * invZoom; // GL-downward default
        (void)width;           // API beibehalten; keine Warnung unter /WX
        return;
    }

    // Fallback: konservative Grundspanne über Breite ableiten.
    constexpr double kBaseSpan = 8.0 / 3.0;
    const double baseStep = kBaseSpan / (width > 0 ? (double)width : 1.0);
    stepX = baseStep * invZoom;
    stepY = -baseStep * invZoom; // GL-downward default
}

// -----------------------------------------------------------------------------
// GID & Pixel-Offsets (Bildmitte als Ursprung; Pixelmitte = +0.5)
// -----------------------------------------------------------------------------
CAPY_HD uint32_t capy_gid(int x, int y, int w) noexcept
{
    return (uint32_t)(y * w + x);
}

CAPY_HD void capy_pixel_offsets(int px, int py, int w, int h,
                                double& offx, double& offy) noexcept
{
    // Center-of-pixel: (+0.5) und dann Bildmitte (w/2,h/2) abziehen
    offx = (double(px) + 0.5) - 0.5 * double(w);
    offy = (double(py) + 0.5) - 0.5 * double(h);
}

// -----------------------------------------------------------------------------
// Klassische Abbildung (double) - stabil bei moderaten Zooms.
// Maus: FMA reduziert Rundungsfehler a*step + center, bleibt aber single-sum.
// -----------------------------------------------------------------------------
CAPY_HD double2 capy_map_pixel_double(double cx, double cy,
                                      double stepX, double stepY,
                                      int px, int py, int w, int h) noexcept
{
    double offx, offy;
    capy_pixel_offsets(px, py, w, h, offx, offy);
    double2 c;
    c.x = capy_fma(offx, stepX, cx);
    c.y = capy_fma(offy, stepY, cy);
    return c;
}

// -----------------------------------------------------------------------------
// Capybara-Abbildung (Hi/Lo) - numerisch stabil bei tiefen Zooms.
// nutzt capy_map_center_step(cx,cy, stepX,stepY, offx,offy)
// -----------------------------------------------------------------------------
CAPY_HD CapyHiLo2 capy_map_pixel_hilo(double cx, double cy,
                                      double stepX, double stepY,
                                      int px, int py, int w, int h) noexcept
{
    double offx, offy;
    capy_pixel_offsets(px, py, w, h, offx, offy);
    return capy_map_center_step(cx, cy, stepX, stepY, offx, offy);
}

// -----------------------------------------------------------------------------
// Bridging-Helper: befüllt gleichzeitig double2 und CapyHiLo2
// -----------------------------------------------------------------------------
CAPY_HD void capy_map_pixel(double cx, double cy,
                            double stepX, double stepY,
                            int px, int py, int w, int h,
                            double2& c_double, CapyHiLo2& c_hilo) noexcept
{
    c_double = capy_map_pixel_double(cx, cy, stepX, stepY, px, py, w, h);
    c_hilo   = capy_map_pixel_hilo  (cx, cy, stepX, stepY, px, py, w, h);
}

// -----------------------------------------------------------------------------
// Optionale Device-Logs (ASCII-only, rate-limited). Keine Hard-Abhängigkeiten.
// Benötigt CAPY_DEBUG_LOGGING, LUCHS_LOG_DEVICE(msg) und capy_should_log(...).
// -----------------------------------------------------------------------------
#if defined(__CUDA_ARCH__)

// kleine Hilfsfunktionen für ASCII-Append (nur für Device-Logs genutzt)
CAPY_D int capy_append_str(char* dst, int pos, int maxLen, const char* s) {
    if (pos >= maxLen) return pos;
    const int limit = maxLen - 1; // 1 Platz für '\0' reservieren
    while (*s && pos < limit) {
        dst[pos++] = *s++;
    }
    return pos;
}

CAPY_D int capy_append_int(char* dst, int pos, int maxLen, long long v) {
    if (pos >= maxLen) return pos;
    const int limit = maxLen - 1;

    if (v == 0) {
        if (pos < limit) dst[pos++] = '0';
        return pos;
    }
    if (v < 0) {
        if (pos < limit) dst[pos++] = '-';
        v = -v;
    }

    char tmp[32];
    int tpos = 0;
    while (v > 0 && tpos < (int)sizeof(tmp)) {
        int digit = (int)(v % 10);
        tmp[tpos++] = (char)('0' + digit);
        v /= 10;
    }
    while (tpos > 0 && pos < limit) {
        dst[pos++] = tmp[--tpos];
    }
    return pos;
}

CAPY_D void capy_log_map_init_if(uint32_t gid, int px, int py,
                                 double2 cD, const CapyHiLo2& cHL,
                                 int earlyIters)
{
#if defined(CAPY_DEBUG_LOGGING) && defined(LUCHS_LOG_DEVICE)
    // Falls extern kein capy_should_log bereitgestellt ist, wäre das ein Build-Thema.
    if (!capy_should_log(gid, (uint32_t)CAPY_LOG_RATE)) return;

    char msg[192];
    int pos = 0;

    pos = capy_append_str(msg, pos, (int)sizeof(msg), "CAPY map gid=");
    pos = capy_append_int(msg, pos, (int)sizeof(msg), (long long)gid);
    pos = capy_append_str(msg, pos, (int)sizeof(msg), " px=");
    pos = capy_append_int(msg, pos, (int)sizeof(msg), (long long)px);
    pos = capy_append_str(msg, pos, (int)sizeof(msg), " py=");
    pos = capy_append_int(msg, pos, (int)sizeof(msg), (long long)py);
    pos = capy_append_str(msg, pos, (int)sizeof(msg), " early=");
    pos = capy_append_int(msg, pos, (int)sizeof(msg), (long long)earlyIters);

    // aktuell kein Float-Dump (cD/cHL), um ohne snprintf auszukommen
    // aber Parameter bleiben erhalten für zukünftige Erweiterung.
    (void)cD;
    (void)cHL;

    msg[(pos < (int)sizeof(msg)) ? pos : ((int)sizeof(msg) - 1)] = '\0';
    LUCHS_LOG_DEVICE(msg);
#else
    (void)gid; (void)px; (void)py; (void)cD; (void)cHL; (void)earlyIters;
#endif
}

#else // !__CUDA_ARCH__

// Host-Seite: Stub, damit Header überall eingebunden werden kann.
inline void capy_log_map_init_if(uint32_t, int, int, double2, const CapyHiLo2&, int) {}

#endif // __CUDA_ARCH__

#undef CAPY_HD
#undef CAPY_D
