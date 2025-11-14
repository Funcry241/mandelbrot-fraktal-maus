///// Otter: Replikatoren - LinUCB/RLS Implementierung mit Sherman-Morrison, ε-Exploration, Top-k Auswahl.
///// Schneefuchs: Numerik in double, deterministische xorshift32-RNG, ASCII-Logs via LUCHS_LOG_HOST.
///// Maus: Persistenz kleiner Binär-Blobs (A⁻¹,b); harte Clamps für Reward/NaN-Hygiene; keine Fremdabhängigkeiten.
///// Datei: src/ai/otter_bandit.cpp
#include "ai/otter_bandit.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <chrono>

namespace otter {
namespace ai {

// --- kleine Utilities --------------------------------------------------------

static inline float clampf(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

static inline bool is_finite(double v) {
    return std::isfinite(v);
}

// --- OtterBandit: öffentliche API -------------------------------------------

void OtterBandit::init(int dim) {
    if (dim <= 0 || dim > 1024) {
        LUCHS_LOG_HOST("[AOP/LIN] init failed: invalid dim");
        return;
    }
    m_dim = dim;
    m_Ainv.assign((size_t)dim * (size_t)dim, 0.0);
    m_b.assign((size_t)dim, 0.0);
    m_w.assign((size_t)dim, 0.0);

    const double invLambda = (m_params.lambda > 0.0f) ? (1.0 / (double)m_params.lambda) : 1e3; // Fallback
    for (int i = 0; i < dim; ++i) {
        m_Ainv[(size_t)i * (size_t)dim + (size_t)i] = invLambda; // A⁻¹ = (1/λ)·I
    }

    if (m_rngState == 0) m_rngState = 0x9E3779B1u;
    LUCHS_LOG_HOST("[AOP/LIN] init: d=%d alpha=%.3f eps=%.3f lambda=%.2e topK=%d", dim, m_params.alpha, m_params.epsilon, (double)m_params.lambda, m_params.topK);
}

void OtterBandit::configure(const BanditParams& p) {
    m_params = p;
    if (m_params.lambda <= 0.0f) m_params.lambda = 1.0e-6f;
    if (m_params.topK <= 0) m_params.topK = 1;
    if (m_params.epsilon < 0.0f) m_params.epsilon = 0.0f;
    if (m_params.epsilon > 0.5f) m_params.epsilon = 0.5f;
}

void OtterBandit::reset() {
    if (m_dim <= 0) return;
    const double invLambda = (m_params.lambda > 0.0f) ? (1.0 / (double)m_params.lambda) : 1e3;
    std::fill(m_Ainv.begin(), m_Ainv.end(), 0.0);
    std::fill(m_b.begin(), m_b.end(), 0.0);
    std::fill(m_w.begin(), m_w.end(), 0.0);
    for (int i = 0; i < m_dim; ++i) {
        m_Ainv[(size_t)i * (size_t)m_dim + (size_t)i] = invLambda;
    }
    LUCHS_LOG_HOST("[AOP/LIN] reset: d=%d", m_dim);
}

void OtterBandit::set_seed(uint32_t s) {
    m_rngState = (s == 0u) ? 0xC0FFEEu : s;
}

float OtterBandit::predict(const float* x) const {
    if (!is_initialized() || !x) return 0.0f;
    return (float)dot_w(x);
}

float OtterBandit::uncertainty(const float* x) const {
    if (!is_initialized() || !x) return 0.0f;
    double q = quadform_Ainv(x);
    if (!is_finite(q) || q < 0.0) q = 0.0;
    return (float)std::sqrt(q);
}

std::vector<BanditScore> OtterBandit::select_topk(const float* X, int numTiles, int stride) const {
    std::vector<BanditScore> out;
    if (!is_initialized() || !X || numTiles <= 0 || stride != m_dim) return out;

    out.reserve((size_t)numTiles);
    for (int i = 0; i < numTiles; ++i) {
        const float* xi = X + (size_t)i * (size_t)stride;
        double pred = dot_w(xi);
        double u = quadform_Ainv(xi);
        u = (u > 0.0 && is_finite(u)) ? std::sqrt(u) : 0.0;
        double s = pred + (double)m_params.alpha * u;
        BanditScore sc;
        sc.index = i;
        sc.score = (float)s;
        sc.ucb   = (float)(m_params.alpha * u);
        out.push_back(sc);
    }

    const int k = std::min(std::max(m_params.topK, 1), numTiles);
    std::partial_sort(out.begin(),
                      out.begin() + k,
                      out.end(),
                      [](const BanditScore& a, const BanditScore& b){ return a.score > b.score; });

    out.resize((size_t)k);

    if (m_params.epsilon > 0.0f && rng_uniform01() < m_params.epsilon && numTiles > k) {
        int replacement = (int)(rng_uniform01() * (float)numTiles);
        if (replacement >= numTiles) replacement = numTiles - 1;

        bool unique = true;
        for (int i = 0; i < k; ++i) {
            if (out[(size_t)i].index == replacement) { unique = false; break; }
        }
        if (unique) {
            const float* xr = X + (size_t)replacement * (size_t)stride;
            double pred = dot_w(xr);
            double u = quadform_Ainv(xr);
            u = (u > 0.0 && is_finite(u)) ? std::sqrt(u) : 0.0;
            double s = pred + (double)m_params.alpha * u;
            out[(size_t)k - 1] = BanditScore{ replacement, (float)s, (float)(m_params.alpha * u) };
        }
    }

    return out;
}

void OtterBandit::update(const float* x, float rewardIn) {
    if (!is_initialized() || !x) return;

    const float rClamped = clampf(rewardIn, m_params.rewardClampLo, m_params.rewardClampHi);

    bool anyFinite = false;
    for (int i = 0; i < m_dim; ++i) {
        if (std::isfinite((double)x[i])) { anyFinite = true; break; }
    }
    if (!anyFinite) {
        LUCHS_LOG_HOST("[AOP/LIN] update skipped: non-finite x");
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    sm_update_Ainv(x);

    for (int i = 0; i < m_dim; ++i) {
        m_b[(size_t)i] += (double)rClamped * (double)x[i];
    }

    recompute_w();

    auto t1 = std::chrono::high_resolution_clock::now();
    const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    LUCHS_LOG_HOST("[AOP/LIN] update: r=%.4f dt=%.3fms", (double)rClamped, dt_ms);
}

bool OtterBandit::save(const char* path) const {
    if (!is_initialized() || !path) return false;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        LUCHS_LOG_HOST("[AOP/LIN] save failed: cannot open file");
        return false;
    }
    const uint32_t magic = 0x4F42414Eu; // 'OBAN'
    f.write((const char*)&magic, sizeof(magic));
    f.write((const char*)&m_dim, sizeof(m_dim));

    f.write((const char*)m_Ainv.data(), sizeof(double) * m_Ainv.size());
    f.write((const char*)m_b.data(),    sizeof(double) * m_b.size());
    const bool ok = (bool)f;
    LUCHS_LOG_HOST(ok ? "[AOP/LIN] save ok" : "[AOP/LIN] save failed: I/O");
    return ok;
}

bool OtterBandit::load(const char* path) {
    if (!path) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        LUCHS_LOG_HOST("[AOP/LIN] load failed: cannot open file");
        return false;
    }
    uint32_t magic = 0;
    int fileDim = 0;
    f.read((char*)&magic, sizeof(magic));
    f.read((char*)&fileDim, sizeof(fileDim));
    if (magic != 0x4F42414Eu || fileDim <= 0) {
        LUCHS_LOG_HOST("[AOP/LIN] load failed: bad header");
        return false;
    }
    init(fileDim);

    f.read((char*)m_Ainv.data(), sizeof(double) * m_Ainv.size());
    f.read((char*)m_b.data(),    sizeof(double) * m_b.size());
    if (!f) {
        LUCHS_LOG_HOST("[AOP/LIN] load failed: I/O");
        return false;
    }
    recompute_w();
    LUCHS_LOG_HOST("[AOP/LIN] load ok: d=%d", m_dim);
    return true;
}

void OtterBandit::get_weights(std::vector<float>& out) const {
    out.resize((size_t)m_dim);
    for (int i = 0; i < m_dim; ++i) out[(size_t)i] = (float)m_w[(size_t)i];
}

std::string OtterBandit::brief() const {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "[AOP/LIN] d=%d a=%.2f eps=%.2f lam=%.1e k=%d stage=%d",
                  m_dim, m_params.alpha, m_params.epsilon, (double)m_params.lambda, m_params.topK,
                  (int)m_params.stage);
    return std::string(buf);
}

// --- OtterBandit: private Mathe ---------------------------------------------

double OtterBandit::dot_w(const float* x) const {
    const int d = m_dim;
    const double* w = m_w.data();
    double acc = 0.0;
    for (int i = 0; i < d; ++i) {
        acc += (double)x[i] * w[(size_t)i];
    }
    return acc;
}

double OtterBandit::quadform_Ainv(const float* x) const {
    const int d = m_dim;
    const double* A = m_Ainv.data();

    // Kein std::vector in der Hot-Loop: fester Stack-Buffer (max d=1024 durch init-Guard).
    double tmp[1024];

    for (int r = 0; r < d; ++r) {
        const double* Ar = A + (size_t)r * (size_t)d;
        double s = 0.0;
        for (int c = 0; c < d; ++c) {
            s += Ar[(size_t)c] * (double)x[c];
        }
        tmp[(size_t)r] = s;
    }

    double q = 0.0;
    for (int i = 0; i < d; ++i) {
        q += (double)x[i] * tmp[(size_t)i];
    }
    return q;
}

void OtterBandit::sm_update_Ainv(const float* x) {
    const int d = m_dim;
    double denom = 1.0 + quadform_Ainv(x);
    if (!(denom > 0.0) || !is_finite(denom)) {
        const double jitter = 1e-9;
        for (int i = 0; i < d; ++i) {
            m_Ainv[(size_t)i * (size_t)d + (size_t)i] += jitter;
        }
        denom = 1.0 + quadform_Ainv(x);
        if (!(denom > 0.0) || !is_finite(denom)) {
            LUCHS_LOG_HOST("[AOP/LIN] sm_update: bad denom");
            return;
        }
    }

    // Ebenfalls: fester Stack-Buffer statt std::vector pro Update.
    double u[1024];

    for (int r = 0; r < d; ++r) {
        const double* Ar = m_Ainv.data() + (size_t)r * (size_t)d;
        double s = 0.0;
        for (int c = 0; c < d; ++c) {
            s += Ar[(size_t)c] * (double)x[c];
        }
        u[(size_t)r] = s;
    }

    const double scale = 1.0 / denom;
    for (int r = 0; r < d; ++r) {
        double* Ar = m_Ainv.data() + (size_t)r * (size_t)d;
        const double ur = u[(size_t)r];
        for (int c = 0; c < d; ++c) {
            Ar[(size_t)c] -= ur * u[(size_t)c] * scale;
        }
    }
}

void OtterBandit::recompute_w() {
    const int d = m_dim;
    const double* A = m_Ainv.data();
    const double* b = m_b.data();
    for (int r = 0; r < d; ++r) {
        const double* Ar = A + (size_t)r * (size_t)d;
        double s = 0.0;
        for (int c = 0; c < d; ++c) s += Ar[(size_t)c] * b[(size_t)c];
        m_w[(size_t)r] = s;
    }
}

// --- RNG ---------------------------------------------------------------------

uint32_t OtterBandit::rng_next() const {
    uint32_t x = m_rngState;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    m_rngState = (x == 0u) ? 0xA5A5A5A5u : x;
    return m_rngState;
}

float OtterBandit::rng_uniform01() const {
    const uint32_t v = rng_next();
    const float f = (float)((v >> 8) & 0x00FFFFFFu) / (float)0x01000000u;
    return (f <= 0.0f) ? 1.0f / 16777216.0f : (f >= 1.0f ? (16777215.0f / 16777216.0f) : f);
}

} // namespace ai
} // namespace otter
