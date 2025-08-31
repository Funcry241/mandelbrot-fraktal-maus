///// Otter: Progressive PI-Controller (header-only)
///// Schneefuchs: Konservativ, stabil, deterministisch.
///// Maus: Ziel ~160 ms; Klemmer und Clamps inklusive.
///// Datei: src/progressive_controller.hpp

#pragma once
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace prog {

struct PISettings {
    float    targetMs   = 160.0f; // Framezeit-Ziel
    float    kP         = 0.22f;  // Proportional
    float    kI         = 0.06f;  // Integral (langsam)
    uint32_t minIter    = 64;     // Untergrenze
    uint32_t maxIter    = 4096;   // Obergrenze
    uint32_t quantum    = 16;     // Schrittgranularität (Rundung)
};

class ProgressivePI {
public:
    explicit ProgressivePI(const PISettings& s = PISettings()) : cfg_(s) {}

    uint32_t suggest(uint32_t currentChunk, float lastMs) {
        // Fehler positiv → schneller (mehr Iterationen), negativ → langsamer
        const float err = (cfg_.targetMs - lastMs);
        integ_ = std::clamp(integ_ + err * 0.5f, -4.0f * cfg_.targetMs, 4.0f * cfg_.targetMs);
        const float delta = cfg_.kP * err + cfg_.kI * integ_;

        // Skaliere Änderung relativ zum Ziel (robust gegen absolute Größen)
        float factor = 1.0f + (delta / std::max(cfg_.targetMs, 1.0f));
        if      (factor < 0.5f) factor = 0.5f;
        else if (factor > 2.0f) factor = 2.0f;

        uint32_t next = (uint32_t)std::lround((double)currentChunk * (double)factor);
        next = std::clamp(next, cfg_.minIter, cfg_.maxIter);
        // Auf Quantum runden
        next = (next / cfg_.quantum) * cfg_.quantum;
        if (next == 0) next = cfg_.minIter;
        return next;
    }

    const PISettings& settings() const { return cfg_; }

private:
    PISettings cfg_;
    float integ_ = 0.0f;
};

} // namespace prog
