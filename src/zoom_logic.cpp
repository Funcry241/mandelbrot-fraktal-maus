///// Otter: Rullmolder Step 2 - blunt zoom + gentle nudge toward Interest (dt-invariant).
///// Schneefuchs: Minimal invasive; caps & deadzone; /WX clean; safe casts (no ref-casts).
///// Maus: Stable ASCII keys; rate-limited; pch first; optional logs [ZPAN1]/[ZPERF]/[ZLEASH]/[ZJIT]/[ZANGL]/[ZDEF]/[ZPILOT]/[ZKEY].
///// Datei: src/zoom_logic.cpp

#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced formal parameter (API preserved)

#include "pch.hpp"
#include "zoom_logic.hpp"
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "axolotel_coupler.hpp" // << Axolotel Zoom-Coupler (boost multiplier)

#include <vector>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <chrono>

// GLFW für Tastatur-Polling (WASD/Arrows); PCH bleibt zuerst.
#include <GLFW/glfw3.h>

#define RS_OFFSET_X(ctx) ((ctx).center.x)
#define RS_OFFSET_Y(ctx) ((ctx).center.y)
#define RS_ZOOM(ctx)     ((ctx).zoom)

namespace ZoomLogic {

// --- tiny helpers (fast math; no transcendentals in hot-path) ----------------

static inline float get_dt_seconds(const FrameContext& fc) noexcept {
    return (fc.deltaSeconds > 0.0f) ? fc.deltaSeconds : (1.0f / 60.0f);
}

// exp(x) with |x| << 1  →  1 + x + x²/2 (error O(x³)); dt≈1/60 → x≈0.003.. ok
static inline double exp_fast2(double x) noexcept {
    return 1.0 + x * (1.0 + 0.5 * x);
}

// 1 - exp(-a)  Padé(1,1) ~ a / (1 + a/2), stable & clampable for a≥0
static inline double one_minus_expm_fast(double a) noexcept {
    const double d = 1.0 + 0.5 * a;
    return (d > 0.0) ? (a / d) : 0.0;
}

static inline double blunt_zoom_rate_per_sec() noexcept {
    return 0.20; // +20%/s
}

// Gentle nudge tunables (slightly stronger to ensure visible effect)
struct NudgeCfg {
    double gainPerSec      = 1.30;
    double deadzoneNdc     = 0.10;
    double maxPxPerFrame   = 16.0;
    double yScale          = 0.94;
    double strengthFloor   = 0.30;
};
static constexpr NudgeCfg kNudge{};

// Axis-weighted Leash: X früh/stark bremsen, Y spät/schwach
struct AxisLeashCfg {
    double xStart = 0.20, xStop = 0.55, xMin = 0.05;
    double yStart = 0.40, yStop = 0.90, yMin = 0.35;
};
static constexpr AxisLeashCfg kLeash{};

// Phase A: Early-Locality Cap (öffnet weich von R0→1.0; nur in diesem TU)
struct StartLeashCfg {
    bool   enabled     = true;
    double R0          = 0.22;
    double openSeconds = 2.4;
    bool   cubicEase   = true;
};
static constexpr StartLeashCfg kStartLeash{};

// ---------------- Keyboard Nav Bias (WASD/Arrows) ----------------------------
// Sanfter Tastatur-Bias in NDC; additiv, dt-invariant integriert.
static double sKeyBiasX = 0.0, sKeyBiasY = 0.0;

static inline void update_key_nav_bias(float dt) noexcept
{
    namespace NB = Settings::NavBias;

    // Halbwert → λ
    const double dtD = (dt > 0.0f) ? static_cast<double>(dt) : 0.0;
    const double lambda = (NB::halfLifeSec > 0.0) ? (std::log(2.0) / NB::halfLifeSec) : 0.0;

    // Zielrichtung u aus Tastenzuständen, normiert (keine √2-Bevorzugung)
    double ux = 0.0, uy = 0.0;

    if constexpr (NB::enabled) {
        if (GLFWwindow* window = glfwGetCurrentContext()) {
            const int xPos = (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
                           + (glfwGetKey(window, GLFW_KEY_D)     == GLFW_PRESS);
            const int xNeg = (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS)
                           + (glfwGetKey(window, GLFW_KEY_A)     == GLFW_PRESS);
            const int yPos = (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS)
                           + (glfwGetKey(window, GLFW_KEY_W)     == GLFW_PRESS);
            const int yNeg = (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS)
                           + (glfwGetKey(window, GLFW_KEY_S)     == GLFW_PRESS);

            const int txi = (xPos > 0) - (xNeg > 0);
            const int tyi = (yPos > 0) - (yNeg > 0);

            const double tx = static_cast<double>(txi);
            const double ty = static_cast<double>(tyi);

            const double L = std::sqrt(tx*tx + ty*ty);
            if (L > 0.0) { 
                ux = tx / L; 
                uy = -(ty / L) * NB::yScale; // Y-Richtung invertiert
            }
        }
    }

    // db/dt = gain*u − λ*b
    sKeyBiasX += dtD * (NB::gainPerSec * ux - lambda * sKeyBiasX);
    sKeyBiasY += dtD * (NB::gainPerSec * uy - lambda * sKeyBiasY);

    // radialer Cap (compile-time)
    if constexpr (NB::maxNdc > 0.0) {
        const double cap  = NB::maxNdc;
        const double m2   = sKeyBiasX*sKeyBiasX + sKeyBiasY*sKeyBiasY;
        const double cap2 = cap*cap;
        if (m2 > cap2) {
            const double invM = 1.0 / std::sqrt(m2);
            const double s    = cap * invM;
            sKeyBiasX *= s; sKeyBiasY *= s;
        }
    }

    // Snap-To-Zero gegen Jitter/Log-Spam
    if (std::abs(sKeyBiasX) < 1e-6) sKeyBiasX = 0.0;
    if (std::abs(sKeyBiasY) < 1e-6) sKeyBiasY = 0.0;
}

static inline void add_key_bias_to_ndc(double& x, double& y) noexcept
{
    if constexpr (Settings::NavBias::enabled) {
        x += sKeyBiasX;
        y += sKeyBiasY;
    }
}

// --- experimental: run-seeded jitter + early deflection + pilot-kick ---------
struct XorShift32 {
    uint32_t s;
    uint32_t next() noexcept {
        if (!s) s = 0xA3C59AC3u;
        s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
    }
    float u01() noexcept { return (next() >> 8) * (1.0f / 16777216.0f); } // [0,1)
};
struct StartNoise {
    bool     seeded        = false;
    bool     jitterDone    = false;
    uint32_t seed          = 0;
    double   angleBiasRad  = 0.0;   // +/- ~24°
    double   angleDurSec   = 2.2;   // fade-out Dauer
    int      deflectSign   = +1;    // +/- 1
    double   deflectMax    = 0.22;  // max orthogonale NDC-Deflektion (stärker)
    double   deflectDurSec = 2.6;   // länger wirksam
    double   pilotMaxPx    = 18.0;  // direkt in Pixel
    double   pilotDurSec   = 1.6;   // kurzer, kräftiger Antritt
    XorShift32 rng{0};
};
static StartNoise sNoise;

static void ensure_seed_once() noexcept {
    if (sNoise.seeded) return;
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    uint64_t mix = static_cast<uint64_t>(now) ^ 0x9E3779B97f4a7c15ULL;
    mix ^= (mix >> 33);
    sNoise.seed = static_cast<uint32_t>(mix ^ (mix >> 32));
    if (!sNoise.seed) sNoise.seed = 0x9E3779B9u;
    sNoise.rng.s = sNoise.seed;

    const double degToRad = 0.017453292519943295;
    const double a = (sNoise.rng.u01() * 2.0 - 1.0) * (24.0 * degToRad);
    sNoise.angleBiasRad = a;
    sNoise.deflectSign  = (sNoise.rng.u01() < 0.5f) ? -1 : +1;

    if constexpr (Settings::ZoomLog::enabled) {
        LUCHS_LOG_HOST("[ZSEED] runSeed=0x%08X angleBias=%.3f deg deflectSign=%+d durA=%.2fs durD=%.2fs pilot=%.1fpx/%.1fs",
                       (unsigned)sNoise.seed, a / degToRad, sNoise.deflectSign,
                       sNoise.angleDurSec, sNoise.deflectDurSec, sNoise.pilotMaxPx, sNoise.pilotDurSec);
    }
    sNoise.seeded = true;
}

// --- local telemetry state ---------------------------------------------------

struct ZLogState {
    uint64_t frame = 0;
    bool     headerPrinted = false;
    double   sinceStartSec = 0.0;
};
static ZLogState zls;

// --- public surface ----------------------------------------------------------

ZoomResult evaluateTarget(const std::vector<float>& /*entropy*/,
                          const std::vector<float>& /*contrast*/,
                          int /*tilesX*/, int /*tilesY*/,
                          int /*width*/, int /*height*/,
                          float2 currentOffset, float /*zoom*/,
                          float2 /*previousOffset*/,
                          ZoomState& state) noexcept
{
    ZoomResult zr{};
    state.hadCandidate = false;
    zr.shouldZoom      = true;
    zr.bestIndex       = -1;
    zr.newOffsetX      = currentOffset.x;
    zr.newOffsetY      = currentOffset.y;
    return zr;
}

// --- small, inlinable helpers (avoid lambdas) --------------------------------

static inline double applyDeadzone(double v, double dz) noexcept {
    const double a = (v >= 0.0) ? v : -v;
    if (a <= dz) return 0.0;
    const double t = (a >= 1.0) ? 1.0 : (a - dz) / (1.0 - dz);
    return (v < 0.0) ? -t : t;
}

static inline double smooth01(double x, double a, double b) noexcept {
    if (x <= a) return 0.0;
    if (x >= b) return 1.0;
    const double t = (x - a) / (b - a);
    return t * t * (3.0 - 2.0 * t);
}

static inline double leashAxis(double a, double s, double e, double minF) noexcept {
    const double r = (a >= 0.0) ? a : -a;
    const double f = 1.0 - smooth01(r, s, e);
    return (f < minF) ? minF : f;
}

static inline double clamp_abs(double v, double cap) noexcept {
    if (v >  cap) return cap;
    if (v < -cap) return -cap;
    return v;
}

// --- core --------------------------------------------------------------------

static void update(FrameContext& frameCtx, RendererState& rs, ZoomState& /*zs*/)
{
    using Clock = std::chrono::steady_clock;
    [[maybe_unused]] const auto tUpdateStart = Clock::now();
    long long pan_us = 0;

    zls.frame++;

    const float  dt   = get_dt_seconds(frameCtx);

    // Base zoom rate and Axolotel coupler boost (multiplier ≥ 1.0)
    double       rate = blunt_zoom_rate_per_sec();
    const float  cplBoost = AxolotelCoupler::boost();
    rate *= static_cast<double>(cplBoost);

    // Laufzeit fürs Startverhalten
    zls.sinceStartSec += static_cast<double>(dt);

    // einmalig: Seed + Startjitter (px→world / zoom)
    if (zls.frame == 1) {
        ensure_seed_once();

        const double rpx   = 22.0 + 24.0 * (double)sNoise.rng.u01();
        const double phi   = 6.283185307179586 * (double)sNoise.rng.u01();
        const double jx_px = rpx * std::cos(phi);
        const double jy_px = rpx * std::sin(phi);

        const double psx = static_cast<double>(rs.pixelScale.x);
        const double psy = static_cast<double>(rs.pixelScale.y);
        const double z   = static_cast<double>(RS_ZOOM(rs));
        const bool   ok  = (psx != 0.0 || psy != 0.0) && (z != 0.0);

        if (ok) {
            const double invZ = 1.0 / z;
            const double dWorldX = jx_px * psx * invZ;
            const double dWorldY = jy_px * psy * invZ;
            RS_OFFSET_X(rs) += dWorldX;
            RS_OFFSET_Y(rs) += dWorldY;
            sNoise.jitterDone = true;
            if constexpr (Settings::ZoomLog::enabled) {
                LUCHS_LOG_HOST("[ZJIT] seed=0x%08X rpx=%.2f phi=%.2f dWorld=(%.9f,%.9f) invZ=%.6g",
                               (unsigned)sNoise.seed, rpx, phi, dWorldX, dWorldY, invZ);
            }
        }
    }

    using ZoomT = std::remove_cv_t<std::remove_reference_t<decltype(RS_ZOOM(rs))>>;
    const ZoomT  z0  = static_cast<ZoomT>(RS_ZOOM(rs));

    // Effective per-frame logarithmic delta with coupler boost
    const double ldz = rate * static_cast<double>(dt);

    // Fast exp: g = exp(ldz) ≈ 1 + ldz + 0.5*ldz^2
    const double g   = exp_fast2(ldz);
    const ZoomT  z1  = static_cast<ZoomT>(static_cast<double>(z0) * g);
    RS_ZOOM(rs) = z1;

    // Logging cadence
    const uint64_t modN       = (Settings::ZoomLog::everyN > 0)
                              ? static_cast<uint64_t>(Settings::ZoomLog::everyN) : 1ULL;
    const bool     emitEveryN = ((zls.frame % modN) == 0);

    // -------------------- Gentle Nudge (PAN) ---------------------------------
    if (rs.interest.valid && rs.width > 0 && rs.height > 0) {
        [[maybe_unused]] const auto tPanStart = Clock::now();

        const double ndcX_raw0 = rs.interest.ndcX;
        const double ndcY_raw0 = rs.interest.ndcY;

        // Early angle bias (±24°, fadet in ~2.2 s auf 0)
        double ndcX_in = ndcX_raw0, ndcY_in = ndcY_raw0;
        if (sNoise.seeded && sNoise.angleBiasRad != 0.0 && sNoise.angleDurSec > 0.0) {
            const double t = std::clamp(1.0 - (zls.sinceStartSec / sNoise.angleDurSec), 0.0, 1.0);
            if (t > 0.0) {
                const double ang = sNoise.angleBiasRad * t;
                const double c = std::cos(ang), s = std::sin(ang);
                const double rx = ndcX_in * c - ndcY_in * s;
                const double ry = ndcX_in * s + ndcY_in * c;
                ndcX_in = rx; ndcY_in = ry;
                if constexpr (Settings::ZoomLog::enabled) {
                    if (emitEveryN) {
                        LUCHS_LOG_HOST("[ZANGL] f=%llu fade=%.2f ang=%.3f ndcRot=(%.3f,%.3f)",
                                       (unsigned long long)zls.frame, t, ang, ndcX_in, ndcY_in);
                    }
                }
            }
        }

        // Orthogonale Deflektion – stark, aber ausfaded
        if (sNoise.seeded && sNoise.deflectMax > 0.0 && sNoise.deflectDurSec > 0.0) {
            const double t = std::clamp(1.0 - (zls.sinceStartSec / sNoise.deflectDurSec), 0.0, 1.0);
            if (t > 0.0) {
                const double r2 = ndcX_in*ndcX_in + ndcY_in*ndcY_in;
                if (r2 > 1e-16) {
                    const double invLen = 1.0 / std::sqrt(r2);
                    const double ox = -ndcY_in * invLen; // 90° links
                    const double oy =  ndcX_in * invLen;
                    const double amp = sNoise.deflectMax * t;
                    ndcX_in += (double)sNoise.deflectSign * amp * ox;
                    ndcY_in += (double)sNoise.deflectSign * amp * oy;

                    if constexpr (Settings::ZoomLog::enabled) {
                        if (emitEveryN) {
                            LUCHS_LOG_HOST("[ZDEF] f=%llu fade=%.2f amp=%.3f sign=%+d ndcDef=(%.3f,%.3f)",
                                           (unsigned long long)zls.frame, t, amp, sNoise.deflectSign, ndcX_in, ndcY_in);
                        }
                    }
                }
            }
        }

        // --- Keyboard Nav Bias (additiv) -------------------------------------
        update_key_nav_bias(dt);
        add_key_bias_to_ndc(ndcX_in, ndcY_in);
        if constexpr (Settings::ZoomLog::enabled) {
            if (emitEveryN && (sKeyBiasX != 0.0 || sKeyBiasY != 0.0)) {
                LUCHS_LOG_HOST("[ZKEY] f=%llu keyBias=(%.4f,%.4f) ndc+key=(%.4f,%.4f)",
                               (unsigned long long)zls.frame, sKeyBiasX, sKeyBiasY, ndcX_in, ndcY_in);
            }
        }

        // Pilot-Kick: kurzer Seitenimpuls in Pixeln
        double pilot_px_x = 0.0, pilot_px_y = 0.0;
        if (sNoise.seeded && sNoise.pilotMaxPx > 0.0 && sNoise.pilotDurSec > 0.0) {
            const double t = std::clamp(1.0 - (zls.sinceStartSec / sNoise.pilotDurSec), 0.0, 1.0);
            if (t > 0.0) {
                const double f = t * t * (3.0 - 2.0 * t);
                const double ampPx = sNoise.pilotMaxPx * f;

                double ox = 0.0, oy = 0.0;
                const double r2 = ndcX_in*ndcX_in + ndcY_in*ndcY_in;
                if (r2 > 1e-12) {
                    const double invLen = 1.0 / std::sqrt(r2);
                    ox = -ndcY_in * invLen;
                    oy =  ndcX_in * invLen;
                } else {
                    const double phi = 6.283185307179586 * (double)sNoise.rng.u01();
                    ox = std::cos(phi); oy = std::sin(phi);
                }
                pilot_px_x = (double)sNoise.deflectSign * ampPx * ox;
                pilot_px_y = (double)sNoise.deflectSign * ampPx * oy;

                if constexpr (Settings::ZoomLog::enabled) {
                    if (emitEveryN) {
                        LUCHS_LOG_HOST("[ZPILOT] f=%llu fade=%.2f ampPx=%.2f dir=(%.3f,%.3f) pilotPx=(%.2f,%.2f)",
                                       (unsigned long long)zls.frame, t, ampPx, ox, oy, pilot_px_x, pilot_px_y);
                    }
                }
            }
        }

        // Deadzone & Leashes
        double ndcX = applyDeadzone(ndcX_in, kNudge.deadzoneNdc);
        double ndcY = applyDeadzone(ndcY_in, kNudge.deadzoneNdc);

        // Early-Locality Cap (öffnet weich von R0 → 1.0)
        if constexpr (kStartLeash.enabled) {
            const double T = (kStartLeash.openSeconds > 0.0) ? kStartLeash.openSeconds : 0.0;
            double t = (T > 0.0) ? std::min(1.0, zls.sinceStartSec / T) : 1.0;
            if constexpr (kStartLeash.cubicEase) t = t * t * (3.0 - 2.0 * t);

            const double R0   = std::clamp(kStartLeash.R0, 0.0, 1.0);
            const double Rcap = R0 + (1.0 - R0) * t;
            const double r2   = ndcX*ndcX + ndcY*ndcY;
            const double R2   = Rcap * Rcap;
            if (r2 > R2 && r2 > 1e-16) {
                const double invR = Rcap / std::sqrt(r2);
                ndcX *= invR;
                ndcY *= invR;
                if constexpr (Settings::ZoomLog::enabled) {
                    if (emitEveryN) {
                        LUCHS_LOG_HOST("[ZLEASH] f=%llu earlyLocality R=%.3f ndc'=(%.3f,%.3f)",
                                       (unsigned long long)zls.frame, Rcap, ndcX, ndcY);
                    }
                }
            }
        }

        // Axis-weighted radial leash
        const double leashX = leashAxis(ndcX, kLeash.xStart, kLeash.xStop, kLeash.xMin);
        const double leashY = leashAxis(ndcY, kLeash.yStart, kLeash.yStop, kLeash.yMin);
        ndcX *= leashX;
        ndcY *= leashY;

        if constexpr (Settings::ZoomLog::enabled) {
            if (emitEveryN && (leashX < 0.999 || leashY < 0.999)) {
                LUCHS_LOG_HOST("[ZLEASH] f=%llu leashX=%.2f leashY=%.2f ndc'=(%.3f,%.3f)",
                               (unsigned long long)zls.frame, leashX, leashY, ndcX, ndcY);
            }
        }

        const bool hitDZ_X = (std::abs(ndcX_in) <= kNudge.deadzoneNdc);
        const bool hitDZ_Y = (std::abs(ndcY_in) <= kNudge.deadzoneNdc);

        if (ndcX != 0.0 || ndcY != 0.0 || (pilot_px_x != 0.0 || pilot_px_y != 0.0)) {
            const double s = std::max(kNudge.strengthFloor, std::min(1.0, rs.interest.strength));

            const double halfW = 0.5 * static_cast<double>(rs.width);
            const double halfH = 0.5 * static_cast<double>(rs.height);
            double dx_px_goal = ndcX * halfW + pilot_px_x;
            double dy_px_goal = ndcY * halfH + pilot_px_y;

            const double a = kNudge.gainPerSec * s * static_cast<double>(dt);
            const double alpha = std::min(1.0, std::max(0.0, one_minus_expm_fast(a)));

            double step_px_x = clamp_abs(dx_px_goal * alpha, kNudge.maxPxPerFrame);
            double step_px_y = clamp_abs(dy_px_goal * alpha * kNudge.yScale, kNudge.maxPxPerFrame);

            const bool hitCAP_X = (step_px_x != dx_px_goal * alpha);
            const bool hitCAP_Y = (step_px_y != dy_px_goal * alpha * kNudge.yScale);

            const double psx = static_cast<double>(rs.pixelScale.x);
            const double psy = static_cast<double>(rs.pixelScale.y);
            const bool scaleZero = (psx == 0.0 && psy == 0.0);

            if (!scaleZero) {
                const double z = static_cast<double>(RS_ZOOM(rs));
                const double invZ = (z != 0.0) ? (1.0 / z) : 0.0;
                const double dWorldX = step_px_x * psx * invZ;
                const double dWorldY = step_px_y * psy * invZ;

                RS_OFFSET_X(rs) += dWorldX;
                RS_OFFSET_Y(rs) += dWorldY;

                const int flags = (hitDZ_X ? 1 : 0)
                                | (hitDZ_Y ? 2 : 0)
                                | (hitCAP_X ? 4 : 0)
                                | (hitCAP_Y ? 8 : 0)
                                | (scaleZero ? 16 : 0);

                if constexpr (Settings::ZoomLog::enabled) {
                    if (emitEveryN) {
                        LUCHS_LOG_HOST("[ZPAN1] f=%llu ndc=(%.4f,%.4f) a=%.3f s=%.2f "
                                       "goal_px=(%.2f,%.2f) step_px=(%.2f,%.2f) dWorld=(%.9f,%.9f) invZ=%.6g flags=0x%02X",
                                       (unsigned long long)zls.frame,
                                       ndcX_in, ndcY_in, alpha, s,
                                       dx_px_goal, dy_px_goal, step_px_x, step_px_y,
                                       dWorldX, dWorldY, invZ, flags);
                    }
                }
            }

            pan_us += (long long)std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tPanStart).count();
        }
    }

    // --- Foundational Zoom Telemetry -----------------------------------------
    if constexpr (Settings::ZoomLog::enabled) {
        const bool needHeader = (Settings::ZoomLog::header && !zls.headerPrinted);
        if (needHeader) {
            LUCHS_LOG_HOST("[ZHDR] keys=f,dt_ms,z0,z1,g,rps,ldz%s",
                           Settings::ZoomLog::includeCenter ? ",cx,cy" : "");
            zls.headerPrinted = true;
        }

        if (emitEveryN) {
            const double dt_ms = static_cast<double>(dt) * 1000.0;
            if (Settings::ZoomLog::includeCenter) {
                const double cx = static_cast<double>(RS_OFFSET_X(rs));
                const double cy = static_cast<double>(RS_OFFSET_Y(rs));
                LUCHS_LOG_HOST("[ZLOG][S2] f=%llu dt_ms=%.3f z0=%.6f z1=%.6f g=%.6f rps=%.6f ldz=%.6f cpl=%.3f cx=%.9f cy=%.9f",
                               (unsigned long long)zls.frame, dt_ms,
                               static_cast<double>(z0), static_cast<double>(z1),
                               g, rate, ldz, static_cast<double>(cplBoost), cx, cy);
            } else {
                LUCHS_LOG_HOST("[ZLOG][S2] f=%llu dt_ms=%.3f z0=%.6f z1=%.6f g=%.6f rps=%.6f ldz=%.6f cpl=%.3f",
                               (unsigned long long)zls.frame, dt_ms,
                               static_cast<double>(z0), static_cast<double>(z1),
                               g, rate, ldz, static_cast<double>(cplBoost));
            }

            if (rs.interest.valid) {
                LUCHS_LOG_HOST("[ZPAN0] f=%llu interest ndc=(%.6f,%.6f) R=%.4f s=%.2f",
                               (unsigned long long)zls.frame,
                               rs.interest.ndcX, rs.interest.ndcY,
                               rs.interest.radiusNdc, rs.interest.strength);
            }

            const long long update_us =
                (long long)std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tUpdateStart).count();
            LUCHS_LOG_HOST("[ZPERF] f=%llu update_us=%lld pan_us=%lld",
                           (unsigned long long)zls.frame, update_us, pan_us);
        }
    }
}

void evaluateAndApply(FrameContext& frameCtx, RendererState& rs, ZoomState& zs, float dtOverrideSeconds) noexcept
{
    const float savedDt = frameCtx.deltaSeconds;
    if (dtOverrideSeconds > 0.0f) frameCtx.deltaSeconds = dtOverrideSeconds;

    update(frameCtx, rs, zs);

    frameCtx.deltaSeconds = savedDt;
}

} // namespace ZoomLogic

#pragma warning(pop)
