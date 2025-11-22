///// Otter: Nacktmull — frame pipeline with Axolotel Coupler; draw-lag-1; pan works during Pause (zoom frozen).
///// Schneefuchs: ASCII logs; pch first; deterministic diffs; [REPL/*] centralized logging.
///// Maus: Compute → Metrics → Overlays → ASM HUD → Axolotel → Zoom; if Pause: interest cleared, zoom restored to pre-call.
///// Datei: src/frame_pipeline.cpp

#include "pch.hpp"
#include <GLFW/glfw3.h>       // glfwGetTime()
#include <chrono>
#include <algorithm>
#include <cstdio>             // snprintf for dynamic ring logging
#include <cmath>              // sqrt
#include <cuda_runtime.h>     // CUDA event timing

#include "capybara_mapping.cuh" // computeTileSizeFromZoom(...)
#include "renderer_resources.hpp"
#include "renderer_pipeline.hpp"
#include "cuda_interop.hpp"
#include "frame_context.hpp"
#include "frame_pipeline.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "hud_text.hpp"
#include "zoom_logic.hpp"
#include "common.hpp"
#include "fps_meter.hpp"
#include "axolotel_hud.hpp"    // ✨ Axolotel WOW-HUD (additive, on key-pulse)
#include "dachs_hud.hpp"       // Dachs-HUD help state/text"
#include "renderer_state.hpp"  // <-- benötigt: vollständige Definition von RendererState
#include "asm/asm_hud_probe.hpp"   // ASM mini-grid for HUD panel (independent of CUDA main render)
#include "asm/asm_hud_panel.hpp"   // ASM mini-panel draw (GL only, uses RendererState::asmHudGrid)

// --- Replikatoren ---------------------------
#include "ai/aop_controller.hpp"  // [REPL/POLICY]
#include "ai/aop_telemetry.hpp"   // [REPL/COUPLE] telemetry (positions/valid)
#include "perturb_core.hpp"       // [REPL/ORBIT]

#include <vector_types.h>
#include <vector_functions.h>

static_assert(Settings::pboRingSize == RendererState::kPboRingSize, "pboRingSize must match Settings::pboRingSize");

// ------------------------------ TU-lokaler Zustand ----------------------------
static FrameContext         g_ctx;
static ZoomLogic::ZoomState g_zoomState;
static int                  g_frame = 0;

namespace {
    using Clock = std::chrono::high_resolution_clock;
    using msd   = std::chrono::duration<double, std::milli>;

    // Nacktmull: cadence driven by Settings::PerfLog
    constexpr int   RING_LOG_EVERY     = 120;
    constexpr float AXO_ZOOM_BOOST_PCT = 0.15f;  // β: +15% dt at E=1
    static   bool   g_forceMetricsNext  = false; // eager metrics trigger

    static double g_mandMs = 0.0;
    static double g_entMs  = 0.0;
    static double g_conMs  = 0.0;
    static double g_texMs  = 0.0;
    static double g_ovlMs  = 0.0;
    static double g_totMs  = 0.0;

    // --- NEW: stale-forwarding for metrics + age counter --------------------
    static int   g_metricsAge = 0;   // frames since last metrics compute
    static float g_lastE0     = 0.f; // last known entropy[0]
    static float g_lastC0     = 0.f; // last known contrast[0]
    static bool  g_haveLast   = false;

    inline bool perfShouldLog(int frameIdx) {
        if constexpr (Settings::performanceLogging) {
            if (!Settings::PerfLog::enabled) return false;
            if (frameIdx <= Settings::PerfLog::warmupFrames) return false;
            return (frameIdx % Settings::PerfLog::everyN) == 0;
        } else {
            (void)frameIdx;
            return false;
        }
    }

    // --- NEW: zusätzlicher Gate NUR für die LANGE [PERF]-Zeile --------------
    // Ziel: Auch wenn PerfLog::everyN klein ist (z.B. 1), wird die *lange* Zeile
    // höchstens alle ~60 Frames ausgegeben - ODER sofort bei signifikanter Änderung
    // (Resolution, Iterations, statsPx).
    inline bool longPerfShouldLog(int frameIdx, const FrameContext& fctx) {
        struct Sig { int w, h, it, statsPx; };
        static bool s_init = false;
        static Sig  s_last{0,0,0,0};
        static int  s_lastEmitFrame = -1000000000;

        const int statsPxCur = std::max(1, fctx.statsTileSize);
        const Sig cur{ fctx.width, fctx.height, fctx.maxIterations, statsPxCur };

        const bool sigChanged =
            (!s_init) ||
            (cur.w != s_last.w) || (cur.h != s_last.h) ||
            (cur.it != s_last.it) || (cur.statsPx != s_last.statsPx);

        // Zeit-Gate: mindestens alle 60 Frames (≈ 1 s @ 60 FPS), unabhängig von Settings
        const int baseN   = std::max(1, Settings::PerfLog::everyN);
        const int minN    = 60;                    // harte Unterkante für die lange Zeile
        const int everyN  = std::max(minN, baseN); // falls baseN größer ist, respektieren

        const bool timeGate = (frameIdx - s_lastEmitFrame) >= everyN;

        if (sigChanged || timeGate) {
            s_last = cur;
            s_init = true;
            s_lastEmitFrame = frameIdx;
            return true;
        }
        return false;
    }

    inline long long epochMillisNow() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }

    static void beginFrameLocal() {
        static double s_prevNow = 0.0;
        const double now = glfwGetTime();
        double dt = (s_prevNow > 0.0) ? (now - s_prevNow) : (1.0 / 60.0);
        s_prevNow = now;

        // clamp dt (gegen Hänger/Breakpoints)
        if (dt < 1.0/300.0) dt = 1.0/300.0;
        if (dt > 1.0/15.0)  dt = 1.0/15.0;

        g_ctx.deltaSeconds = static_cast<float>(dt);

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPE] beginFrame: time=%.4f, dt=%.5f, totalFrames=%d", now, dt, g_frame);
        }
        ++g_frame;
    }

    // ---------------------- Host-Fallback für Metrics ----------------------
    static void ensureHeatmapHostData(RendererState& state, int width, int height, int tilePx) {
        const int px = std::max(1, tilePx);
        const int tx = (width  + px - 1) / px;
        const int ty = (height + px - 1) / px;
        const size_t N = static_cast<size_t>(tx) * static_cast<size_t>(ty);

        const bool needEnt = state.h_entropy.size()  != N || state.h_entropy.empty();
        const bool needCon = state.h_contrast.size() != N || state.h_contrast.empty();
        if (!needEnt && !needCon) return;

        if (needEnt) state.h_entropy.assign(N, 0.0f);
        if (needCon) state.h_contrast.assign(N, 0.0f);

        for (int y = 0; y < ty; ++y) {
            for (int x = 0; x < tx; ++x) {
                const size_t i = static_cast<size_t>(y) * tx + x;
                const float fx = (tx > 1) ? (float)x / (float)(tx - 1) : 0.0f;
                const float fy = (ty > 1) ? (float)y / (float)(tx - 1) : 0.0f;
                const float r  = std::min(1.0f, std::sqrt(fx*fx + fy*fy));
                state.h_entropy[i]  = 0.15f + 0.8f * r;

                const int  checker = ((x ^ y) & 1);
                const float mix    = 0.3f + 0.7f * ((fx + (1.0f - fy)) * 0.5f);
                state.h_contrast[i] = checker ? mix : (1.0f - mix);
            }
        }

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[HM][FALLBACK] generated N=%zu tiles=%dx%d tilePx=%d", N, tx, ty, px);
        }
    }

    // ---------------------- Metrics EINMAL pro Frame ------------------------
    static void ensureAnalysisMetrics(FrameContext& fctx, RendererState& state)
    {
        const int statsPx = std::max(1,
            (Settings::Kolibri::gridScreenConstant ? Settings::Kolibri::desiredTilePx : fctx.tileSize));

        const bool needBootstrap = state.h_entropy.empty() || state.h_contrast.empty();

        bool forceNow = g_forceMetricsNext;
        if (forceNow) g_forceMetricsNext = false;

        const bool shouldCompute =
            needBootstrap ||
            forceNow ||
            ((g_frame % Settings::StatsCadence::heatmapEveryN) == 0);

        if (!shouldCompute) {
            // No device work, no host syncs: reuse last metrics verbatim.
            fctx.statsTileSize = statsPx;
            fctx.entropy       = state.h_entropy;
            fctx.contrast      = state.h_contrast;

            if constexpr (Settings::performanceLogging) {
                g_entMs = 0.0;
                g_conMs = 0.0;
            }
            // age++ on reuse
            ++g_metricsAge;

            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[HM][SKIP] reuse metrics frame=%d everyN=%d age=%d",
                               g_frame, Settings::StatsCadence::heatmapEveryN, g_metricsAge);
            }
            return;
        }

        bool ok = false;
        if constexpr (Settings::performanceLogging) {
            cudaEvent_t evM0 = nullptr, evM1 = nullptr;
            (void)cudaEventCreateWithFlags(&evM0, cudaEventDefault);
            (void)cudaEventCreateWithFlags(&evM1, cudaEventDefault);
            (void)cudaEventRecord(evM0, state.renderStream);

            ok = CudaInterop::buildHeatmapMetrics(state, fctx.width, fctx.height, statsPx, state.renderStream);

            (void)cudaEventRecord(evM1, state.renderStream);
            (void)cudaEventSynchronize(evM1);
            float msMetrics = 0.0f;
            (void)cudaEventElapsedTime(&msMetrics, evM0, evM1);
            g_entMs = (double)msMetrics;  // combined metrics duration
            g_conMs = 0.0;
            (void)cudaEventDestroy(evM0);
            (void)cudaEventDestroy(evM1);
        } else {
            ok = CudaInterop::buildHeatmapMetrics(state, fctx.width, fctx.height, statsPx, state.renderStream);
        }

        if (!ok || state.h_entropy.empty() || state.h_contrast.empty()) {
            ensureHeatmapHostData(state, fctx.width, fctx.height, statsPx);
        }

        // In den FrameContext spiegeln (Decoupling!)
        fctx.statsTileSize = statsPx;
        fctx.entropy       = state.h_entropy;
        fctx.contrast      = state.h_contrast;

        // Reset age & capture last known e0/c0
        g_metricsAge = 0;
        if (!state.h_entropy.empty()) { g_lastE0 = state.h_entropy[0]; g_haveLast = true; }
        if (!state.h_contrast.empty()) { g_lastC0 = state.h_contrast[0]; g_haveLast = true; }

        if constexpr (Settings::performanceLogging) {
            const int compPx = std::max(1, fctx.tileSize);
            const int ovTx   = (fctx.width  + statsPx - 1) / statsPx;
            const int ovTy   = (fctx.height + statsPx - 1) / statsPx;
            const int cTx    = (fctx.width  + compPx  - 1) / compPx;
            const int cTy    = (fctx.height + compPx  - 1) / compPx;
            LUCHS_LOG_HOST("[GRID] statsPx=%d stats=%dx%d computePx=%d compute=%dx%d res=%dx%d",
                           statsPx, ovTx, ovTy, compPx, cTx, cTy, fctx.width, fctx.height);

            // ---- Sanity line for metrics payload -----------------------------
            if (!state.h_entropy.empty() && !state.h_contrast.empty()) {
                auto mmE = std::minmax_element(state.h_entropy.begin(),  state.h_entropy.end());
                auto mmC = std::minmax_element(state.h_contrast.begin(), state.h_contrast.end());
                const float eMin = *mmE.first;
                const float eMax = *mmE.second;
                const float cMin = *mmC.first;
                const float cMax = *mmC.second;
                LUCHS_LOG_HOST("[HM][VERIFY] N=%zu statsPx=%d E[min=%.4f max=%.4f] C[min=%.4f max=%.4f]",
                               state.h_entropy.size(), statsPx, eMin, eMax, cMin, cMax);
            } else {
                LUCHS_LOG_HOST("[HM][VERIFY] N=0 statsPx=%d E[min=0.0000 max=0.0000] C[min=0.0000 max=0.0000]", statsPx);
            }
            // ------------------------------------------------------------------
        }
    }

    // ------------------------------- CUDA (Compute) -------------------------------
    static void computeCudaFrame(FrameContext& fctx, RendererState& state) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPE] compute begin: tile=%d it=%d zoom=%.6f",
                           fctx.tileSize, fctx.maxIterations, (double)fctx.zoom);
        }

        // Keep the computed tile size (no unconditional full-res fallback).
        FrameContext fctxRender = fctx;

        if constexpr (Settings::performanceLogging) {
            cudaEvent_t evStart = nullptr, evStop = nullptr;
            (void)cudaEventCreateWithFlags(&evStart, cudaEventDefault);
            (void)cudaEventCreateWithFlags(&evStop,  cudaEventDefault);
            (void)cudaEventRecord(evStart, state.renderStream);

            CudaInterop::renderCudaFrame(state, fctxRender, fctx.newOffsetD.x, fctx.newOffsetD.y);

            (void)cudaEventRecord(evStop, state.renderStream);
            (void)cudaEventSynchronize(evStop);
            float ms = 0.0f;
            (void)cudaEventElapsedTime(&ms, evStart, evStop);
            g_mandMs = (double)ms;
            (void)cudaEventDestroy(evStart);
            (void)cudaEventDestroy(evStop);
        } else {
            CudaInterop::renderCudaFrame(state, fctxRender, fctx.newOffsetD.x, fctx.newOffsetD.y);
        }

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPE] compute end");
        }

        // Upload -> aktuelle Upload-Textur
        const auto t0 = Clock::now();
        if (!state.skipUploadThisFrame) {
            OpenGLUtils::updateTextureFromPBO(state.currentPBO().id(),
                                              state.currentUploadTex().id(),
                                              fctx.width, fctx.height);
            if (state.pboFence[state.pboIndex]) {
                glDeleteSync(state.pboFence[state.pboIndex]);
                state.pboFence[state.pboIndex] = 0;
            }
            state.pboFence[state.pboIndex] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZK][UP] fence set pbo=%u ring=%d", state.currentPBO().id(), state.pboIndex);
            }
        } else {
            state.skipUploadThisFrame = false;
            ++state.ringSkip;
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ZK][UP] skip upload pbo=%u ring=%d", state.currentPBO().id(), state.pboIndex);
            }
        }

        // 🔁 Saubere PBO-Ring-Disziplin: immer weiterschalten
        state.advancePboRing();

        const auto tUploadEnd = Clock::now();
        g_texMs = std::chrono::duration_cast<msd>(tUploadEnd - t0).count();

        // Draw -> die vorherige (fertige) Draw-Textur
        RendererPipeline::drawFullscreenQuad(state.currentDrawTex().id());

        // Nach dem Draw: Upload-Textur wird zur neuen Draw-Textur
        state.advanceTexRingAfterDraw();
    }

    // ------------------------------- Overlays ------------------------------------
    static void drawOverlays(RendererState& state, const FrameContext& fctx) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fctx.width, fctx.height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_STENCIL_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        const auto tOv0 = Clock::now();

        // Overlays lesen (falls noetig) die Draw-Textur (ID direkt weiterreichen)
        const unsigned drawTexId = state.currentDrawTex().id();

        if (state.heatmapOverlayEnabled) {
            const int overlayTilePx = std::max(1, (fctx.statsTileSize > 0 ? fctx.statsTileSize : fctx.tileSize));

            if constexpr (Settings::performanceLogging) {
                const int compPx = std::max(1, fctx.tileSize);
                const int ovTx   = (fctx.width  + overlayTilePx - 1) / overlayTilePx;
                const int ovTy   = (fctx.height + overlayTilePx - 1) / overlayTilePx;
                const int compTx = (fctx.width  + compPx - 1) / compPx;
                const int compTy = (fctx.height + compPx - 1) / compPx;
                LUCHS_LOG_HOST("[GRID] overlayPx=%d overlay=%dx%d computePx=%d compute=%dx%d res=%dx%d",
                               overlayTilePx, ovTx, ovTy, compPx, compTx, compTy, fctx.width, fctx.height);
            }

            HeatmapOverlay::drawOverlay(state.h_entropy, state.h_contrast,
                                        fctx.width, fctx.height, overlayTilePx,
                                        drawTexId, state);
        }

        // Warzenschwein-HUD:
        if constexpr (Settings::warzenschweinOverlayEnabled) {
            if (!DachsHUD::help_enabled()) {
                state.warzenschweinText = HudText::build(fctx, state);
                WarzenschweinOverlay::setText(state.warzenschweinText);
            } else {
                WarzenschweinOverlay::setText("");
            }
            WarzenschweinOverlay::drawOverlay(fctx.zoom);
        }

        // ASM Mini-Fraktal-Panel (rechts unten, eigenes ASM-Grid)
        AsmHudPanel::draw(state, fctx.width, fctx.height);

        // ✨ Axolotel: additive Glow-Pulse (liegt über Panel & Heatmap)
        AxolotelHUD::draw(fctx.width, fctx.height, glfwGetTime());

        const auto tOv1 = Clock::now();
        g_ovlMs = std::chrono::duration_cast<msd>(tOv1 - tOv0).count();
        state.lastTimings.overlaysMs = g_ovlMs;

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPE] overlays end: ovMs=%.3f", g_ovlMs);
        }
    }

    // -------------------------- Compute-Tile-Alignment ---------------------------
    inline int chooseComputeTileSize(float zoom) {
        int t = computeTileSizeFromZoom(zoom);
        t = std::clamp(t, Settings::MIN_TILE_SIZE, Settings::MAX_TILE_SIZE);
        if (t % 32 != 0) {
            const int up   = ((t + 31) / 32) * 32;
            const int down = (t / 32) * 32;
            int cand = (std::abs(up - t) < std::abs(t - down)) ? up : down;
            if (cand < 32) cand = 32;
            t = cand;
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[GRID] align compute tile from %d to %d", (int)computeTileSizeFromZoom(zoom), t);
            }
        }
        return t;
    }
} // anon ns

// ============================================================================

namespace FramePipeline {

void execute(RendererState& state) {
    const auto tFrame0 = Clock::now();

    beginFrameLocal();

    // Interest zu Framebeginn invalidieren - wird vom HeatmapOverlay bei Bedarf gesetzt
    state.interest.valid = false;

    // ---- Autoritative Double-Werte aus dem RendererState ----
    g_ctx.width         = state.width;
    g_ctx.height        = state.height;
    g_ctx.maxIterations = state.maxIterations;
    g_ctx.zoomD         = state.zoom;
    g_ctx.offsetD       = { state.center.x, state.center.y };
    g_ctx.newOffsetD    = g_ctx.offsetD;
    g_ctx.syncFloatFromDouble();

    // Compute-Raster (Kernel) - nur für Logs/Overlays relevant
    g_ctx.tileSize = chooseComputeTileSize(g_ctx.zoom);

    if constexpr (Settings::Kolibri::gridScreenConstant) {
        static int s_prevOverlayPx = -1;
        const int overlayPx = Settings::Kolibri::desiredTilePx;
        if constexpr (Settings::performanceLogging) {
            if (s_prevOverlayPx != overlayPx) {
                const int px = std::max(1, overlayPx);
                const int ts = std::max(1, g_ctx.tileSize);

                const int ovTx = (g_ctx.width  + px - 1) / px;
                const int ovTy = (g_ctx.height + px - 1) / px;
                const int cTx  = (g_ctx.width  + ts - 1) / ts;
                const int cTy  = (g_ctx.height + ts - 1) / ts;

                LUCHS_LOG_HOST("[GRID] overlayPx=%d overlay=%dx%d computePx=%d tiles=%dx%d res=%dx%d",
                               px, ovTx, ovTy, ts, cTx, cTy, g_ctx.width, g_ctx.height);
                s_prevOverlayPx = overlayPx;
            }
        }
    }

    // ---- Replikatoren: Orbit-Gate vor Compute ----
    Repl::Orbit::maybe_prepare_orbit(g_ctx, state);

    // ---- Render (CUDA) ----
    computeCudaFrame(g_ctx, state);

    // ---- Analysis-Metrics (Cadence-Guard) ----
    ensureAnalysisMetrics(g_ctx, state);

    // ---- ASM HUD probe grid (same grid-res as heatmap metrics) -------------
    {
        const int tilePx = std::max(1, (g_ctx.statsTileSize > 0 ? g_ctx.statsTileSize : g_ctx.tileSize));
        const int tilesX = (g_ctx.width  + tilePx - 1) / tilePx;
        const int tilesY = (g_ctx.height + tilePx - 1) / tilePx;

        if (tilesX > 0 && tilesY > 0 && state.maxIterations > 0) {
            state.asmHudTilesX = tilesX;
            state.asmHudTilesY = tilesY;
            state.asmHudGrid = asm_hud_probe::buildHudProbeGrid(
                state,
                g_ctx,
                tilesX,
                tilesY,
                state.maxIterations
            );
        } else {
            state.asmHudTilesX = 0;
            state.asmHudTilesY = 0;
            state.asmHudGrid.clear();
        }
    }

    // ---- Replikatoren: Policy nach Metrics ----
    if constexpr (Settings::Ai::enabled && Settings::Ai::aopEnabled) {
        auto d = Repl::Policy::evaluate_tile_policy(g_ctx, state);
        (void)d; // Entscheidungen folgen in Phase 2
    }

    // ---- Stage 2: AUTO retarget (hart, mit Takt & Guards) -------------------
    if constexpr (Settings::Ai::enabled) {
        if constexpr (Settings::AiBandit::stage == 2) {
            // Nur bei frischen Metrics und wenn Overlay/Policy gültig.
            if (AOP_Telemetry::g_ai_ov_valid && g_metricsAge == 0 && !DachsHUD::help_enabled()) {
                static int s_nextAllowFrame = 0; // Retarget-Takt (Hysterese)
                if (g_frame >= s_nextAllowFrame) {
                    const double ndcAx = static_cast<double>(AOP_Telemetry::g_ai_ndc_pol_x);
                    const double ndcAy = static_cast<double>(AOP_Telemetry::g_ai_ndc_pol_y);
                    // Clamp in [-1,1] zur Sicherheit.
                    const auto clamp_ndc = [](double v) {
                        return std::max(-1.0, std::min(1.0, v));
                    };
                    state.interest.ndcX = clamp_ndc(ndcAx);
                    state.interest.ndcY = clamp_ndc(ndcAy);
                    state.interest.valid = true;
                    g_forceMetricsNext = true; // eager fresh metrics after retarget

                    s_nextAllowFrame = g_frame + std::max(1, Settings::AiBandit::retargetInterval);

                    if constexpr (Settings::performanceLogging) {
                        LUCHS_LOG_HOST("[REPL/AUTO] retarget ndc=(%.3f,%.3f) next=%d",
                                       (float)state.interest.ndcX, (float)state.interest.ndcY, s_nextAllowFrame);
                    }
                } else {
                    if constexpr (Settings::performanceLogging) {
                        const int left = s_nextAllowFrame - g_frame;
                        LUCHS_LOG_HOST("[REPL/AUTO] skip-lock framesLeft=%d", left);
                    }
                }
            }
        }
    }

    // ---- REPL/COUPLE Telemetrie (nur Log, keine Verhaltensänderung) --------
    if (perfShouldLog(g_frame)) {
        const int fresh = (g_metricsAge == 0) ? 1 : 0;
        const int valid = (AOP_Telemetry::g_ai_ov_valid ? 1 : 0);
        LUCHS_LOG_HOST(
            "[REPL/COUPLE] d=%.3f ndc=(%.3f,%.3f) ov=(%.3f,%.3f) hmAge=%d fresh=%d valid=%d",
            AOP_Telemetry::g_ai_last_delta,
            AOP_Telemetry::g_ai_ndc_pol_x, AOP_Telemetry::g_ai_ndc_pol_y,
            AOP_Telemetry::g_ai_ndc_ovl_x, AOP_Telemetry::g_ai_ndc_ovl_y,
            g_metricsAge, fresh, valid
        );
    }

    // ---- Overlays (nutzen die vorliegenden Metrics + ASM-Grid) ----
    drawOverlays(state, g_ctx);

    // ---- Axolotel Coupler -> Zoom (ein Pfad: dt-Scaling) -------------------
    float E = AxolotelHUD::activityEnergy(); // 0..1 from live pulses
    float dtScaled = g_ctx.deltaSeconds;
    if (E > 0.0f) {
        dtScaled = g_ctx.deltaSeconds * (1.0f + AXO_ZOOM_BOOST_PCT * std::clamp(E, 0.0f, 1.0f));
        g_forceMetricsNext = true; // eager metrics next frame for snappy overlays
        if constexpr (Settings::performanceLogging) {
            const float pct = (dtScaled / std::max(1e-6f, g_ctx.deltaSeconds) - 1.0f) * 100.0f;
            LUCHS_LOG_HOST("[AXO][COUPLER] E=%.3f dt=%.4f -> %.4f (+%.1f%%)", E, g_ctx.deltaSeconds, dtScaled, pct);
        }
    }

    // ---- AI Soft-Coupling into interest (gentle NDC blend) -------------------
    if constexpr (Settings::Ai::enabled) {
        if (Settings::Ai::coupleEnabled &&
            AOP_Telemetry::g_ai_ov_valid &&
            g_metricsAge == 0) // nur mit frischen Metrics koppeln
        {
            const double ndcAx = static_cast<double>(AOP_Telemetry::g_ai_ndc_pol_x);
            const double ndcAy = static_cast<double>(AOP_Telemetry::g_ai_ndc_pol_y);

            if (!state.interest.valid) {
                state.interest.ndcX = ndcAx;
                state.interest.ndcY = ndcAy;
                state.interest.valid = true;
                if constexpr (Settings::performanceLogging) {
                    if (perfShouldLog(g_frame)) {
                        LUCHS_LOG_HOST("[AI/BLEND] adopt ndc=(%.3f,%.3f)", (float)ndcAx, (float)ndcAy);
                    }
                }
            } else {
                const double alpha = std::clamp((double)Settings::Ai::hintBlend, 0.0, 1.0);
                const double inX = state.interest.ndcX;
                const double inY = state.interest.ndcY;
                state.interest.ndcX = inX * (1.0 - alpha) + ndcAx * alpha;
                state.interest.ndcY = inY * (1.0 - alpha) + ndcAy * alpha;
                if constexpr (Settings::performanceLogging) {
                    if (perfShouldLog(g_frame)) {
                        LUCHS_LOG_HOST("[AI/BLEND] alpha=%.2f ndc=(%.3f,%.3f)->(%.3f,%.3f)",
                                       (float)alpha, (float)inX, (float)inY,
                                       (float)state.interest.ndcX, (float)state.interest.ndcY);
                    }
                }
            }
        }
    }

    // ---- Zoom/Pan Anwendung -------------------------------------------------
    const bool paused = CudaInterop::getPauseZoom();

    // Falls pausiert: Autopilot-Pan verhindern → Interest invalidieren.
    if (paused) {
        state.interest.valid = false;
    }

    // evaluateAndApply IMMER ausführen:
    //  - bei Pause: Pilot-Override-Pan erlaubt (Interest invalid), Zoom wird nachher zurückgesetzt.
    //  - sonst: normales Verhalten.
    const double preZoom = (double)state.zoom;
    ZoomLogic::evaluateAndApply(g_ctx, state, g_zoomState, /*dtOverrideSeconds*/ dtScaled);

    // Zoom in Pause einfrieren (Pan bleibt erhalten, s.o.)
    if (paused) {
        state.zoom = (float)preZoom;
    }

    // Spiegel zurück in den Context (für HUD/Nächsten Frame)
    g_ctx.offsetD = { state.center.x, state.center.y };
    g_ctx.zoomD   = state.zoom;
    g_ctx.syncFloatFromDouble();

    const auto tFrame1 = Clock::now();
    g_totMs = std::chrono::duration_cast<msd>(tFrame1 - tFrame0).count();
    state.lastTimings.frameTotalMs = g_totMs;

    // HUD: FPS Meter füttern
    FpsMeter::updateCoreMs(g_totMs);

    // --- LANGE [PERF]-Zeile jetzt *doppelt* gegated: PerfLog-Gate UND Long-Gate
    if (perfShouldLog(g_frame) && longPerfShouldLog(g_frame, g_ctx)) {
        const long long tEpoch = epochMillisNow();
        const int resX = g_ctx.width, resY = g_ctx.height;
        const int it   = g_ctx.maxIterations;
        const double fps    = (g_totMs > 1e-3) ? (1000.0 / g_totMs) : 0.0;
        const double maxfps = (g_texMs > 1e-3) ? (1000.0 / g_texMs) : 0.0;

        // --- NEW: stale-forward e0/c0 + age marker --------------------------
        const float e0 = !state.h_entropy.empty()  ? state.h_entropy[0]
                        : (g_haveLast ? g_lastE0 : 0.f);
        const float c0 = !state.h_contrast.empty() ? state.h_contrast[0]
                        : (g_haveLast ? g_lastC0 : 0.f);

        const int   ringIx = state.pboIndex;
        const unsigned pbo = state.currentPBO().id();
        const unsigned tex = state.currentDrawTex().id();

        const size_t hmN = state.h_entropy.size();
        const int statsPx = std::max(1, g_ctx.statsTileSize);

        // --- other-time (oth) schließt Budget zu tot (geclamped >= 0) ---
        double oth = g_totMs - (g_mandMs + g_entMs + g_conMs + g_texMs + g_ovlMs);
        if (oth < 0.0) {
            if (oth > -0.01) oth = 0.0;
        }

        char line[740];
        const int n = std::snprintf(
            line, sizeof(line),
            "[PERF] t=%lld frame=%d res=%dx%d zoom=%.6f it=%d fps=%.2f maxfps=%.2f "
            "mand=%.2f ent=%.2f con=%.2f up=%.2f ovl=%.2f oth=%.2f tot=%.2f "
            "e0=%.4f c0=%.4f hmAge=%d ring=%d skip=%d pbo=%u tex=%u hmN=%zu statsPx=%d",
            tEpoch, g_frame, resX, resY, (double)g_ctx.zoom, it, fps, maxfps,
            g_mandMs, g_entMs, g_conMs, g_texMs, g_ovlMs, oth, g_totMs,
            e0, c0, g_metricsAge, ringIx, (int)state.skipUploadThisFrame, pbo, tex, hmN, statsPx
        );
        line[(n >= 0 && n < (int)sizeof(line)) ? n : (int)sizeof(line) - 1] = '\0';
        LUCHS_LOG_HOST("%s", line);
    }

    if constexpr (Settings::performanceLogging) {
        if ((g_frame % RING_LOG_EVERY) == 0) {
            char buf[256]; int pos = 0;
            pos += std::snprintf(buf + pos, sizeof(buf) - pos, "{");
            for (int i = 0; i < RendererState::kPboRingSize; ++i) {
                pos += std::snprintf(buf + pos, sizeof(buf) - pos, (i == 0 ? "%u" : ",%u"), state.ringUse[i]);
            }
            std::snprintf(buf + pos, sizeof(buf) - pos, "}");
            LUCHS_LOG_HOST("[RING] use=%s skip=%u size=%d", buf, state.ringSkip, RendererState::kPboRingSize);
            for (int i = 0; i < RendererState::kPboRingSize; ++i) state.ringUse[i] = 0;
            state.ringSkip = 0;
        }
    }
}

} // namespace FramePipeline
