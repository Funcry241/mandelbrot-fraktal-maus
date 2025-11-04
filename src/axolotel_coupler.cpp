///// Otter: Axolotel Zoom-Coupler — exponential attack/release smoother + gain -> zoom boost multiplier.
///// Schneefuchs: No hot-path allocs; ASCII logs; branchless clamps; zero GL dependencies.
///// Maus: attackMs/decayMs tunable; perfLog optional; safe defaults; integrates with renderer_loop tick().
///// Datei: src/axolotel_coupler.cpp

#include "pch.hpp"
#include "axolotel_coupler.hpp"
#include "axolotel_hud.hpp"
#include "settings_axolotel.hpp"
#include "luchs_log_host.hpp"

#include <algorithm>
#include <cmath>

namespace AxolotelCoupler {

namespace {
    // Internal smoothed state
    static float sE      = 0.0f;  // smoothed energy (0..1)
    static float sBoost  = 1.0f;  // 1 .. 1+gain
    static bool  sInit   = false;
    static bool  sEnable = Settings::Axolotel::couplerEnabled;

    inline float clamp01(float v){ return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

    inline float smoothStep(float curr, float target, float alpha){
        // one-pole smoother; alpha in [0..1]
        return curr + (target - curr) * clamp01(alpha);
    }

    inline float alphaFromTau(float dt, float tauSec){
        // Convert time constant to one-pole alpha; numerically safe for small dt.
        if (tauSec <= 1e-6f) return 1.0f;
        const float x = -dt / tauSec;
        // 1 - exp(-dt/tau)
        return 1.0f - std::exp(x);
    }

    void ensureInit(){
        if (sInit) return;
        sInit = true;
        sE = 0.0f;
        sBoost = 1.0f;
        sEnable = Settings::Axolotel::couplerEnabled;
        if constexpr (Settings::Axolotel::couplerPerfLog) {
            LUCHS_LOG_HOST("[AXO-CPL] init enabled=%d gain=%.3f atk=%.0fms dcy=%.0fms",
                           sEnable ? 1 : 0,
                           Settings::Axolotel::couplerGain,
                           Settings::Axolotel::couplerRiseMs,
                           Settings::Axolotel::couplerDecayMs);
        }
    }
} // anon

void init(){ ensureInit(); }

void shutdown(){
    sInit   = false;
    sE      = 0.0f;
    sBoost  = 1.0f;
    // keep sEnable as-is; external toggle may persist preference
    if constexpr (Settings::Axolotel::couplerPerfLog) {
        LUCHS_LOG_HOST("[AXO-CPL] shutdown");
    }
}

void setEnabled(bool enabled){ sEnable = enabled; }
bool isEnabled(){ return sEnable; }

void tick(float dtSeconds){
    ensureInit();

    if (!sEnable || !Settings::Axolotel::enabled) {
        // hard bypass
        sE = 0.0f;
        sBoost = 1.0f;
        return;
    }

    if (dtSeconds < 0.f) dtSeconds = 0.f;

    // 1) Sample instantaneous energy from HUD (cheap O(pulses))
    const float eInstant = AxolotelHUD::activityEnergy(); // 0..1
    const float eTarget  = clamp01(eInstant);

    // 2) Exponential smoothing with separate attack/decay
    const float tauRise  = std::max(0.001f, Settings::Axolotel::couplerRiseMs  / 1000.0f);
    const float tauDecay = std::max(0.001f, Settings::Axolotel::couplerDecayMs / 1000.0f);
    const float aRise    = alphaFromTau(dtSeconds, tauRise);
    const float aDecay   = alphaFromTau(dtSeconds, tauDecay);

    const bool rising = (eTarget > sE);
    const float alpha = rising ? aRise : aDecay;
    sE = smoothStep(sE, eTarget, alpha);

    // 3) Map to multiplicative zoom boost
    const float g = std::max(0.0f, Settings::Axolotel::couplerGain);
    const float targetBoost = 1.0f + g * sE;
    // Apply the *same* alpha as energy (feels cohesive)
    sBoost = smoothStep(sBoost, targetBoost, alpha);

    if constexpr (Settings::Axolotel::couplerPerfLog) {
        if (sE > 0.0f) {
            LUCHS_LOG_HOST("[AXO-CPL] e=%.3f boost=%.3f dt=%.3f a=%.3f", sE, sBoost, dtSeconds, alpha);
        }
    }
}

float energy(){ ensureInit(); return clamp01(sE); }
float boost() { ensureInit(); return (sEnable ? sBoost : 1.0f); }

} // namespace AxolotelCoupler
