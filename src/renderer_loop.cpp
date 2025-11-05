///// Otter: Main loop; Silk-Lite motion + frame budget pacing; Axolotel key-pulse + Zoom-Coupler tick; Dachs-HUD toggle + Reset/Copy-State.
///// Schneefuchs: Device/host logs getrennt; flush bei CUDA-Fehlerpfaden; ASCII-only; Keybinds: F1(help), R(reset), Space(pause), Ctrl+C(copy), ESC(quit).
///// Maus: Kein Screenshot/Capture-Feature mehr; deterministische Pfade; /WX clean.
///// Datei: src/renderer_loop.cpp

#include "pch.hpp"
#include "renderer_loop.hpp"
#include "frame_pipeline.hpp"
#include "cuda_interop.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "heatmap_overlay.hpp"
#include "frame_limiter.hpp"
#include "warzenschwein_overlay.hpp"
#include "axolotel_hud.hpp"
#include "axolotel_coupler.hpp"
#include "dachs_hud.hpp"
#include <cuda_runtime_api.h>

#include <cstdio>
#include <string>
#include <filesystem>
#include <cstring>

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
#endif

namespace RendererLoop {

namespace {
    constexpr int PERF_LOG_EVERY = 30;

    inline void beginFrameLocal(RendererState& state) {
        const double now = glfwGetTime();
        double delta = now - state.lastTime;
        if (delta < 0.0) delta = 0.0;
        state.deltaTime = static_cast<float>(delta < 0.001 ? 0.001f : static_cast<float>(delta));
        state.lastTime  = now;
        state.frameCount++;
    }

    inline void initVSyncOnce() {
        static bool vsyncInit = false;
        if (vsyncInit) return;
        vsyncInit = true;
        if constexpr (Settings::preferVSync) {
            glfwSwapInterval(1);
            if constexpr (Settings::performanceLogging) LUCHS_LOG_HOST("[VSync] swapInterval=1");
        } else {
            glfwSwapInterval(0);
            if constexpr (Settings::performanceLogging) LUCHS_LOG_HOST("[VSync] swapInterval=0");
        }
    }

    // Copy-State Helfer (Ctrl+C)
    static bool copy_state_to_clipboard_or_file(const RendererState& s) {
        char line[160];
        std::snprintf(line, sizeof(line), "cx=%.9f cy=%.9f zoom=%.6f",
                      (double)s.center.x, (double)s.center.y, (double)s.zoom);

    #if defined(_WIN32)
        const std::string utf8 = std::string(line);
        int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(), nullptr, 0);
        if (wlen > 0) {
            HGLOBAL hglb = GlobalAlloc(GMEM_MOVEABLE, (SIZE_T)((wlen + 1) * sizeof(wchar_t)));
            if (hglb) {
                wchar_t* wstr = (wchar_t*)GlobalLock(hglb);
                if (wstr) {
                    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int)utf8.size(), wstr, wlen);
                    wstr[wlen] = L'\0';
                    GlobalUnlock(hglb);
                    if (OpenClipboard(nullptr)) {
                        EmptyClipboard();
                        SetClipboardData(CF_UNICODETEXT, hglb);
                        CloseClipboard();
                        return true;
                    }
                }
                GlobalFree(hglb);
            }
        }
        // Fallback: Datei
    #endif
        std::filesystem::create_directories("captures");
        FILE* f = nullptr;
    #if defined(_MSC_VER)
        if (fopen_s(&f, "captures/last_state.txt", "wb") != 0) f = nullptr;
    #else
        f = std::fopen("captures/last_state.txt", "wb");
    #endif
        if (!f) return false;
        const size_t n = std::fwrite(line, 1, std::strlen(line), f);
        std::fclose(f);
        return n == std::strlen(line);
    }
}

void renderFrame_impl(RendererState& state) {
    initVSyncOnce();
    beginFrameLocal(state);

    AxolotelCoupler::tick(state.deltaTime);
    FramePipeline::execute(state);

    if constexpr (Settings::debugLogging) {
        const cudaError_t err = cudaPeekAtLastError();
        const bool periodic = (state.frameCount % PERF_LOG_EVERY) == 0;
        if (err != cudaSuccess || periodic) {
            LUCHS_LOG_HOST("[Loop] flushing device logs (err=%d, frame=%d)",
                           static_cast<int>(err), state.frameCount);
            LuchsLogger::flushDeviceLogToHost(0);
        }
        if (periodic) {
            LUCHS_LOG_HOST("[Loop] frame=%d dt=%.3f", state.frameCount, state.deltaTime);
        }
    }

    static pace::FrameLimiter limiter;
    if constexpr (Settings::capFramerate) {
        limiter.limit(Settings::capTargetFps);
    } else {
        limiter.limit(0);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode;
    if (action != GLFW_PRESS) return;

    auto* state = static_cast<RendererState*>(glfwGetWindowUserPointer(window));
    if (!state) return;

    AxolotelHUD::noteKeyPress(key, mods);

    switch (key) {
        case GLFW_KEY_A: {
            const bool newEnabled = !AxolotelHUD::isEnabled();
            AxolotelHUD::setEnabled(newEnabled);
            AxolotelCoupler::setEnabled(newEnabled);
            if constexpr (Settings::performanceLogging) {
                LUCHS_LOG_HOST("[AXO] toggle enabled=%d", newEnabled ? 1 : 0);
            }
            break;
        }
        case GLFW_KEY_H:
            HeatmapOverlay::toggle(*state);
            break;
        case GLFW_KEY_O:
            WarzenschweinOverlay::toggle();
            break;
        case GLFW_KEY_SPACE: {
            const bool paused = CudaInterop::getPauseZoom();
            CudaInterop::setPauseZoom(!paused);
            break;
        }
        case GLFW_KEY_F1: {
            const bool on = DachsHUD::toggle_help();
            if (on) {
                DachsHUD::set_text(DachsHUD::Pane::Center, DachsHUD::build_help_text());
            }
            LUCHS_LOG_HOST("[HELP] toggled=%d", on ? 1 : 0);
            break;
        }
        case GLFW_KEY_R: {
            state->center.x = Settings::initialOffsetX;
            state->center.y = Settings::initialOffsetY;
            state->zoom     = Settings::initialZoom;
            LUCHS_LOG_HOST("[RESET] cx=%.9f cy=%.9f z=%.6f",
                           (double)state->center.x, (double)state->center.y, (double)state->zoom);
            break;
        }
        case GLFW_KEY_C: {
            if (mods & GLFW_MOD_CONTROL) {
                const bool ok = copy_state_to_clipboard_or_file(*state);
                LUCHS_LOG_HOST("[CLIP] cx=%.9f cy=%.9f z=%.6f ok=%d",
                               (double)state->center.x, (double)state->center.y, (double)state->zoom, ok ? 1 : 0);
            }
            break;
        }
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        default:
            break;
    }
}

} // namespace RendererLoop
