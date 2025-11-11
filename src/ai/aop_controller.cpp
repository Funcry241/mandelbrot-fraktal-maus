///// Otter: AOP controller – Replikatoren Shadow-Preview + Bandit-Scoring (LinUCB), rein on-device, deterministisch.
///// Schneefuchs: Cadence via PerfLog; ASCII-One-Liner; keine Fremdlibs; Stage-Labels (shadow/assist/auto) in allen Policy-Logs.
///// Maus: Nutzt FeaturePacker (NCHW & Bandit-Matrix); sicherer Reward-Proxy erst bei Retarget; Fallback auf z-Score.
///// Datei: src/ai/aop_controller.cpp

#include "pch.hpp"
#include "ai/aop_controller.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "renderer_state.hpp"
#include "frame_context.hpp"
#include "ai/aop_telemetry.hpp"
#include "ai/feature_packer.hpp"
#include "ai/otter_bandit.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cfloat>
#include <vector>

namespace Repl { namespace Policy {

static inline float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// --- Stage-Label für Logs -----------------------------------------------------
static inline const char* stage_name() {
    const int s = Settings::AiBandit::stage;
    return (s == 2) ? "auto" : (s == 1) ? "assist" : "shadow";
}

// --- Statischer Bandit-State -------------------------------------------------
struct ReplBanditState {
    otter::ai::OtterBandit bandit;
    int dim = 0;
    int tilesX = 0;
    int tilesY = 0;
    int lastPick = -1;
    float lastSignal = 0.0f;
    std::vector<float> lastX;
    int lastUpdateFrame = -9999;
    bool ready = false;
};
static ReplBanditState& RBs() {
    static ReplBanditState s;
    return s;
}

// --- Signal ------------------------------------------------------------------
static inline float compute_signal(float e, float c) {
    return e + Settings::AiBandit::beta * c;
}

// --- z-Score Fallback --------------------------------------------------------
static size_t pick_by_zscore(const float* E, const float* C, size_t N) {
    double sumE=0.0,sumC=0.0,sumE2=0.0,sumC2=0.0;
    for (size_t i=0;i<N;++i){ const double e=E[i], c=C[i]; sumE+=e; sumC+=c; sumE2+=e*e; sumC2+=c*c; }
    const double invN = (N>0)?(1.0/(double)N):0.0;
    const double mE=sumE*invN, mC=sumC*invN;
    const double vE=std::max(0.0,sumE2*invN - mE*mE);
    const double vC=std::max(0.0,sumC2*invN - mC*mC);
    const double sE=(vE>0.0)?std::sqrt(vE):1.0;
    const double sC=(vC>0.0)?std::sqrt(vC):1.0;

    size_t best=0; double bestZ=-1e30;
    for (size_t i=0;i<N;++i){
        const double z = (E[i]-mE)/sE + (C[i]-mC)/sC;
        if (z>bestZ){bestZ=z; best=i;}
    }
    return best;
}

Decision evaluate_tile_policy(const FrameContext& fctx, const RendererState& state)
{
    Decision d{};

    if constexpr (!(Settings::performanceLogging && Settings::PerfLog::enabled)) {
        return d;
    }

    const int warm = Settings::PerfLog::warmupFrames;
    const int every= (Settings::PerfLog::everyN > 0) ? Settings::PerfLog::everyN : 1;
    if (!(state.frameCount > warm && (state.frameCount % every) == 0)) {
        return d;
    }

    const int statsPx = std::max(1, (fctx.statsTileSize > 0) ? fctx.statsTileSize : Settings::Kolibri::desiredTilePx);
    if (state.width <= 0 || state.height <= 0) {
        LUCHS_LOG_HOST("[REPL/POLICY] stage=%s dry-run: invalid dims w=%d h=%d", stage_name(), state.width, state.height);
        return d;
    }

    const size_t w = (size_t)state.width, h = (size_t)state.height;
    const size_t tilesX = (w + (size_t)statsPx - 1) / (size_t)statsPx;
    const size_t tilesY = (h + (size_t)statsPx - 1) / (size_t)statsPx;
    const size_t N = std::min(tilesX*tilesY, std::min(state.h_entropy.size(), state.h_contrast.size()));

    if (N == 0 || tilesX == 0 || tilesY == 0 || state.h_entropy.empty() || state.h_contrast.empty()) {
        LUCHS_LOG_HOST("[REPL/POLICY] stage=%s dry-run: no-metrics N=%zu tiles=%zux%zu statsPx=%d",
                       stage_name(), N, tilesX, tilesY, statsPx);
        return d;
    }

    const float* E = state.h_entropy.data();
    const float* C = state.h_contrast.data();

    using namespace Repl::Feat;
    BanditMatrix X = make_bandit_feature_matrix(state.h_entropy, state.h_contrast, (int)w, (int)h, statsPx);

    auto& RB = RBs();
    if (!RB.ready || RB.dim != X.dim) {
        // ✔️ Parametrisierung an Settings angleichen + Seed sauber setzen
        otter::ai::BanditParams bp{};
        bp.alpha            = Settings::AiBandit::alpha;
        bp.epsilon          = Settings::AiBandit::epsilon;
        bp.lambda           = Settings::AiBandit::lambda;
        bp.beta             = Settings::AiBandit::beta;
        bp.topK             = Settings::AiBandit::topK;
        bp.retargetInterval = Settings::AiBandit::retargetInterval;
        bp.rewardClampLo    = Settings::AiBandit::rewardClampLo;
        bp.rewardClampHi    = Settings::AiBandit::rewardClampHi;
        bp.stage            = (otter::ai::BanditStage)Settings::AiBandit::stage;

        RB.bandit = otter::ai::OtterBandit(X.dim, bp);
        RB.bandit.set_seed((uint32_t)Settings::AiBandit::seed);

        RB.dim    = X.dim;
        RB.tilesX = (int)X.tilesX;
        RB.tilesY = (int)X.tilesY;
        RB.ready  = true;
        LUCHS_LOG_HOST("%s", RB.bandit.brief().c_str());
    }

    // Auswahl (Top-k)
    std::vector<otter::ai::BanditScore> picks = RB.bandit.select_topk(X.data.data(), (int)(X.tilesX*X.tilesY), X.stride);

    // Fallback
    if (picks.empty()) {
        const size_t bestZ = pick_by_zscore(E, C, N);
        const size_t txz = (tilesX ? (bestZ % tilesX) : 0);
        const size_t tyz = (tilesX ? (bestZ / tilesX) : 0);
        const float pxX = ((float)txz + 0.5f) * (float)statsPx;
        const float pxY = ((float)tyz + 0.5f) * (float)statsPx;
        const float ndcX = clamp01(pxX / (float)state.width)  * 2.0f - 1.0f;
        const float ndcY = clamp01(pxY / (float)state.height) * 2.0f - 1.0f;

        // Telemetrie
        AOP_Telemetry::g_ai_ndc_pol_x = ndcX;
        AOP_Telemetry::g_ai_ndc_pol_y = ndcY;
        AOP_Telemetry::g_ai_ndc_ovl_x = ndcX * 0.75f;
        AOP_Telemetry::g_ai_ndc_ovl_y = ndcY * 0.75f;
        AOP_Telemetry::g_ai_ov_valid  = 1;
        AOP_Telemetry::g_ai_last_delta = std::sqrt(ndcX*ndcX + ndcY*ndcY);
        ++AOP_Telemetry::g_ai_frame_id;
        AOP_Telemetry::g_ai_confidence = 0.5f; // neutral im Fallback

        LUCHS_LOG_HOST("[REPL/POLICY] stage=%s zscore-fallback tiles=%zux%zu statsPx=%d best=%zu ndc=(%.3f,%.3f)",
                       stage_name(), tilesX, tilesY, statsPx, bestZ, ndcX, ndcY);
        d.order = 1; d.rebase=false; d.enablePerturb=false;
        return d;
    }

    const int best = picks[0].index;
    const size_t tx = (tilesX ? ((size_t)best % tilesX) : 0);
    const size_t ty = (tilesX ? ((size_t)best / tilesX) : 0);

    const float pxX = ((float)tx + 0.5f) * (float)statsPx;
    const float pxY = ((float)ty + 0.5f) * (float)statsPx;
    const float ndcX = clamp01(pxX / (float)state.width)  * 2.0f - 1.0f;
    const float ndcY = clamp01(pxY / (float)state.height) * 2.0f - 1.0f;

    // Telemetrie
    AOP_Telemetry::g_ai_ndc_pol_x = ndcX;
    AOP_Telemetry::g_ai_ndc_pol_y = ndcY;
    AOP_Telemetry::g_ai_ndc_ovl_x = ndcX * 0.75f;
    AOP_Telemetry::g_ai_ndc_ovl_y = ndcY * 0.75f;
    AOP_Telemetry::g_ai_ov_valid  = 1;
    AOP_Telemetry::g_ai_last_delta = std::sqrt(ndcX*ndcX + ndcY*ndcY);
    ++AOP_Telemetry::g_ai_frame_id;
    {
        const float u = picks[0].ucb;
        const float conf = std::fmax(0.0f, std::fmin(1.0f, 0.5f + 0.5f * std::tanh(u)));
        AOP_Telemetry::g_ai_confidence = conf;
    }

    // -------------------- Reward-Proxy & Update (nur alle k Frames) -----------
    const int frame = (int)state.frameCount;
    const bool doUpdate = ((frame - RB.lastUpdateFrame) >= Settings::AiBandit::retargetInterval);

    if (doUpdate) {
        if (RB.lastPick >= 0 && RB.lastPick < (int)N && !RB.lastX.empty()) {
            const float e_now = E[RB.lastPick];
            const float c_now = C[RB.lastPick];
            float r = compute_signal(e_now, c_now) - RB.lastSignal;

            const float lo = Settings::AiBandit::rewardClampLo;
            const float hi = Settings::AiBandit::rewardClampHi;
            if (!(r==r)) r = 0.0f;
            if (r < lo) r = lo; else if (r > hi) r = hi;

            RB.bandit.update(RB.lastX.data(), r);
        }

        // neuen „letzten Pick“ merken (aktuellen)
        RB.lastPick = best;
        RB.lastX.assign(X.data.begin() + (size_t)best * (size_t)X.stride,
                        X.data.begin() + (size_t)(best+1) * (size_t)X.stride);
        const float e_cur = E[best], c_cur = C[best];
        RB.lastSignal = compute_signal(e_cur, c_cur);
        RB.lastUpdateFrame = frame;
        static int s_updateCount = 0; ++s_updateCount;

        if constexpr (Settings::AiBandit::persistEvery > 0) {
            if ((s_updateCount % Settings::AiBandit::persistEvery) == 0) {
                const bool okSave = RB.bandit.save(Settings::AiBandit::persistPath);
                LUCHS_LOG_HOST(okSave ? "[AI/SAVE] path=%s" : "[AI/SAVE] failed path=%s", Settings::AiBandit::persistPath);
            }
        }
    }

    // --- Logline (Stage-spezifisch) ------------------------------------------
    LUCHS_LOG_HOST("[REPL/POLICY] stage=%s tiles=%zux%zu statsPx=%d best=%d score=%.3f ucb=%.3f ndc=(%.3f,%.3f)",
                   stage_name(), tilesX, tilesY, statsPx, best, picks[0].score, picks[0].ucb, ndcX, ndcY);

    d.order = 1;
    d.rebase = false;
    d.enablePerturb = false;
    return d;
}

}} // namespace Repl::Policy
