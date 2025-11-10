///// Otter: AOP controller – Replikatoren Shadow-Preview + Bandit-Scoring (LinUCB), rein on-device, deterministisch.
///// Schneefuchs: Cadence via PerfLog; ASCII-One-Liner; keine Fremdlibs; Dry-Run/Shadow (keine Steuerung), nur Telemetrie.
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

// --- Statischer Bandit-Zustand (nur in diesem TU) ---------------------------
struct ReplBanditState {
    otter::ai::OtterBandit bandit;
    int dim = 0;
    int tilesX = 0, tilesY = 0;
    int lastPick = -1;
    float lastSignal = 0.0f;     // S = E + beta*C am letzten Pick
    std::vector<float> lastX;    // Featurezeile des letzten Picks
    int lastUpdateFrame = -9999;
    bool ready = false;
};

static ReplBanditState& gRB() {
    static ReplBanditState s;
    return s;
}

// --- Hilfen -----------------------------------------------------------------
static inline float compute_signal(float e, float c) {
    return e + Settings::AiBandit::beta * c;
}

// Fallback z-Score (für Shadow-Vergleich)
static size_t pick_by_zscore(const float* E, const float* C, size_t N) {
    double sumE=0.0,sumC=0.0,sumE2=0.0,sumC2=0.0;
    for (size_t i=0;i<N;++i){ const double e=E[i], c=C[i]; sumE+=e; sumC+=c; sumE2+=e*e; sumC2+=c*c; }
    const double invN = (N>0)?(1.0/(double)N):0.0;
    const double mE=sumE*invN, mC=sumC*invN;
    const double vE=std::max(0.0,sumE2*invN - mE*mE);
    const double vC=std::max(0.0,sumC2*invN - mC*mC);
    const double sE=std::sqrt(vE), sC=std::sqrt(vC);

    size_t best=0; float bestS=-FLT_MAX;
    for (size_t i=0;i<N;++i){
        const float zE = (sE>1e-12)? (float)(((double)E[i]-mE)/sE):0.f;
        const float zC = (sC>1e-12)? (float)(((double)C[i]-mC)/sC):0.f;
        const float s  = zE + zC;
        if (s > bestS){ bestS = s; best = i; }
    }
    return best;
}

Decision evaluate_tile_policy(const FrameContext& fctx, const RendererState& state)
{
    Decision d{};

    // Compile-time/logging cadence gate (shadow/preview frequency).
    if constexpr (!(Settings::performanceLogging && Settings::PerfLog::enabled)) {
        return d;
    }

    // Runtime cadence (tie preview to PERF cadence)
    const int warmupFrames = Settings::PerfLog::warmupFrames;
    const int everyN       = (Settings::PerfLog::everyN > 0) ? Settings::PerfLog::everyN : 1;
    if (!(state.frameCount > warmupFrames && (state.frameCount % everyN) == 0)) {
        return d;
    }

    // Grid
    int statsPx_i = fctx.statsTileSize;
    if (statsPx_i <= 0) statsPx_i = Settings::Kolibri::desiredTilePx;
    const int px_i = std::max(1, statsPx_i);

    if (state.width <= 0 || state.height <= 0) {
        LUCHS_LOG_HOST("[REPL/POLICY] dry-run: invalid dims w=%d h=%d", state.width, state.height);
        return d;
    }

    const size_t w  = (size_t)state.width;
    const size_t h  = (size_t)state.height;
    const size_t tilesX = (w + (size_t)px_i - 1) / (size_t)px_i;
    const size_t tilesY = (h + (size_t)px_i - 1) / (size_t)px_i;
    const size_t N  = std::min(tilesX*tilesY,
                        std::min(state.h_entropy.size(), state.h_contrast.size()));

    if (N == 0 || tilesX == 0 || tilesY == 0 ||
        state.h_entropy.empty() || state.h_contrast.empty())
    {
        LUCHS_LOG_HOST("[REPL/POLICY] dry-run: no-metrics N=%zu tiles=%zux%zu statsPx=%d",
                       N, tilesX, tilesY, px_i);
        return d;
    }

    const float* E = state.h_entropy.data();
    const float* C = state.h_contrast.data();

    // ---- Feature packing (Heatmap + Bandit-Matrix) --------------------------
    using namespace Repl::Feat;
    (void)pack_heatmap_features(state.h_entropy, state.h_contrast,
                                (int)w, (int)h, px_i);

    BanditMatrix X = make_bandit_feature_matrix(state.h_entropy, state.h_contrast,
                                                (int)w, (int)h, px_i);

    // --- Bandit initialisieren (einmalig oder bei Dim-Änderung) -------------
    auto& RB = gRB();
    if (!RB.ready || RB.dim != X.dim) {
        RB.bandit = otter::ai::OtterBandit(X.dim, otter::ai::BanditParams{
            Settings::AiBandit::alpha,
            Settings::AiBandit::epsilon,
            Settings::AiBandit::lambda,
            Settings::AiBandit::beta,
            Settings::AiBandit::topK,
            Settings::AiBandit::retargetInterval,
            Settings::AiBandit::rewardClampLo,
            Settings::AiBandit::rewardClampHi,
            /*persist*/false,
            otter::ai::BanditStage::Shadow
        });
        RB.bandit.set_seed(Settings::AiBandit::seed);
        RB.dim = X.dim;
        RB.tilesX = X.tilesX; RB.tilesY = X.tilesY;
        RB.lastPick = -1;
        RB.lastX.clear();
        RB.lastUpdateFrame = -9999;
        RB.ready = true;
        LUCHS_LOG_HOST("%s", RB.bandit.brief().c_str());
    }

    // --- Auswahl per Bandit (Top-k); Shadow-Mode => nur Telemetrie ----------
    std::vector<otter::ai::BanditScore> picks = RB.bandit.select_topk(
        X.data.data(), (int)(X.tilesX * X.tilesY), X.stride);

    // Fallback, falls irgendwas leer ist
    if (picks.empty()) {
        const size_t bestZ = pick_by_zscore(E, C, N);
        const size_t txz = (tilesX ? (bestZ % tilesX) : 0);
        const size_t tyz = (tilesX ? (bestZ / tilesX) : 0);
        const float pxCenterX = ((float)txz + 0.5f) * (float)px_i;
        const float pxCenterY = ((float)tyz + 0.5f) * (float)px_i;
        const float ndcX = clamp01(pxCenterX / (float)state.width)  * 2.0f - 1.0f;
        const float ndcY = clamp01(pxCenterY / (float)state.height) * 2.0f - 1.0f;

        AOP_Telemetry::g_ai_ndc_pol_x = ndcX;
        AOP_Telemetry::g_ai_ndc_pol_y = ndcY;
        AOP_Telemetry::g_ai_ndc_ovl_x = ndcX * 0.75f;
        AOP_Telemetry::g_ai_ndc_ovl_y = ndcY * 0.75f;
        AOP_Telemetry::g_ai_ov_valid  = 1;
        AOP_Telemetry::g_ai_last_delta = std::sqrt(ndcX*ndcX + ndcY*ndcY);
        ++AOP_Telemetry::g_ai_frame_id;

        LUCHS_LOG_HOST("[REPL/POLICY] zscore-fallback tiles=%zux%zu statsPx=%d best=%zu ndc=(%.3f,%.3f)",
                       tilesX, tilesY, px_i, bestZ, ndcX, ndcY);
        d.order = 1; d.rebase=false; d.enablePerturb=false;
        return d;
    }

    const int best = picks[0].index;
    const size_t tx = (tilesX ? ((size_t)best % tilesX) : 0);
    const size_t ty = (tilesX ? ((size_t)best / tilesX) : 0);

    const float pxCenterX = ((float)tx + 0.5f) * (float)px_i;
    const float pxCenterY = ((float)ty + 0.5f) * (float)px_i;
    const float ndcX = clamp01(pxCenterX / (float)state.width)  * 2.0f - 1.0f;
    const float ndcY = clamp01(pxCenterY / (float)state.height) * 2.0f - 1.0f;

    // Telemetrie (Shadow)
    AOP_Telemetry::g_ai_ndc_pol_x = ndcX;
    AOP_Telemetry::g_ai_ndc_pol_y = ndcY;
    AOP_Telemetry::g_ai_ndc_ovl_x = ndcX * 0.75f;
    AOP_Telemetry::g_ai_ndc_ovl_y = ndcY * 0.75f;
    AOP_Telemetry::g_ai_ov_valid  = 1;
    AOP_Telemetry::g_ai_last_delta = std::sqrt(ndcX*ndcX + ndcY*ndcY);
    ++AOP_Telemetry::g_ai_frame_id;

    // -------------------- Reward-Proxy & Update (nur alle k Frames) ---------
    const int frame = (int)state.frameCount;
    const bool doUpdate = ((frame - RB.lastUpdateFrame) >= Settings::AiBandit::retargetInterval);

    if (doUpdate) {
        // Wenn wir schon einen Pick hatten, update mit ΔS
        if (RB.lastPick >= 0 && RB.lastPick < (int)N && !RB.lastX.empty()) {
            const float e_now = E[RB.lastPick];
            const float c_now = C[RB.lastPick];
            const float S_now = compute_signal(e_now, c_now);
            float r = S_now - RB.lastSignal;
            // Clamp/Norm
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
    }

    // --- Logline (Shadow) ----------------------------------------------------
    LUCHS_LOG_HOST("[REPL/POLICY] shadow tiles=%zux%zu statsPx=%d best=%d score=%.3f ucb=%.3f ndc=(%.3f,%.3f) k=%d",
                   tilesX, tilesY, px_i, best, picks[0].score, picks[0].ucb, ndcX, ndcY,
                   (int)picks.size());

    // Shadow: keine echte Steuerung
    d.order = 1;
    d.rebase = false;
    d.enablePerturb = false;
    return d;
}

}} // namespace Repl::Policy
