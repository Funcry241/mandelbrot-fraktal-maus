///// Otter: Renderer-Core – GL init + window; enables filtered KHR_debug (noisy severities off); no zoom logic here.
///// Schneefuchs: Strict CUDA/GL separation; deterministic ASCII logs; resources clearly owned; duplicate resize removed.
///// Maus: Progressive cooldown + Tatze 7 soft-invalidate on view jumps (post-pipeline, no memset).
///// Datei: src/renderer_core.cu

#include "pch.hpp"
#include "luchs_log_host.hpp"   // LUCHS_LOG_HOST
#include <GL/glew.h>            // GLEW + KHR_debug tokens

#include "renderer_core.hpp"
#include "renderer_state.hpp"
#include "renderer_window.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"
#include "cuda_interop.hpp"
#include "heatmap_overlay.hpp"
#include "frame_pipeline.hpp"
#include "settings.hpp"

#include <stdexcept>
#include <algorithm>

namespace {
    void APIENTRY GlDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei,
                                  const GLchar* message, const void*) {
        // Filter: skip noisy severities
        if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
        if (severity == GL_DEBUG_SEVERITY_LOW)          return;

        const char* src =
            source == GL_DEBUG_SOURCE_API             ? "API" :
            source == GL_DEBUG_SOURCE_WINDOW_SYSTEM   ? "WIN" :
            source == GL_DEBUG_SOURCE_SHADER_COMPILER ? "SHDR" :
            source == GL_DEBUG_SOURCE_THIRD_PARTY     ? "3RD" :
            source == GL_DEBUG_SOURCE_APPLICATION     ? "APP" : "OTH";

        const char* typ =
            type == GL_DEBUG_TYPE_ERROR               ? "ERR"  :
            type == GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR ? "DEPR" :
            type == GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR  ? "UNDEF":
            type == GL_DEBUG_TYPE_PORTABILITY         ? "PORT" :
            type == GL_DEBUG_TYPE_PERFORMANCE         ? "PERF" :
            type == GL_DEBUG_TYPE_MARKER              ? "MARK" :
            type == GL_DEBUG_TYPE_PUSH_GROUP          ? "PUSH" :
            type == GL_DEBUG_TYPE_POP_GROUP           ? "POP"  : "OTH";

        const char* sev =
            severity == GL_DEBUG_SEVERITY_HIGH   ? "HIGH" :
            severity == GL_DEBUG_SEVERITY_MEDIUM ? "MED"  : "LOW";

        LUCHS_LOG_HOST("[GLDBG][%s][%s][%s] id=%u msg=\"%s\"", src, typ, sev, id, message);
    }
} // anon

// --- Renderer impl ------------------------------------------------------------

Renderer::Renderer(int w, int h)
: state(w, h) // RendererState hat keinen Default-Konstruktor
{
}

Renderer::~Renderer() = default;

bool Renderer::initGL() {
    // Create window + GL context
    state.window = RendererWindow::createWindow(state.width, state.height, this);
    if (!state.window) {
        throw std::runtime_error("GLFW window init failed");
    }

    // GLEW init (post-context)
    glewExperimental = GL_TRUE;
    GLenum rc = glewInit();
    if (rc != GLEW_OK) {
        LUCHS_LOG_HOST("[INIT][ERR] glewInit rc=%u", (unsigned)rc);
        throw std::runtime_error("GLEW init failed");
    }

    // Enable KHR_debug with filtered severities
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(&GlDebugCallback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_LOW,           0, nullptr, GL_FALSE);

    // SRGB default FB capability (info only) – robust gegen Token-Varianten
    {
        GLint srgbCap = 0;
        bool logged = false;
        #ifdef GL_FRAMEBUFFER_SRGB_CAPABLE
            glGetIntegerv(GL_FRAMEBUFFER_SRGB_CAPABLE, &srgbCap);
            LUCHS_LOG_HOST("[INIT] GL sRGB-capable default FB: %d", (int)srgbCap);
            logged = true;
        #elif defined(GL_FRAMEBUFFER_SRGB_CAPABLE_EXT)
            glGetIntegerv(GL_FRAMEBUFFER_SRGB_CAPABLE_EXT, &srgbCap);
            LUCHS_LOG_HOST("[INIT] GL sRGB-capable default FB (EXT): %d", (int)srgbCap);
            logged = true;
        #endif
        if (!logged) {
            LUCHS_LOG_HOST("[INIT] GL sRGB-capable default FB: token-missing");
        }
    }

    // CUDA context log
    CudaInterop::logCudaDeviceContext("init");

    glInitialized = true;
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[INIT] GL init complete (w=%d h=%d)", state.width, state.height);
    }

    // Prepare GPU pipeline
    RendererPipeline::init();

    // Allocate device buffers (tileSize stays configured in pipeline/zoom logic)
    state.setupCudaBuffers(Settings::BASE_TILE_SIZE > 0 ? Settings::BASE_TILE_SIZE : 16);

    return true;
}

bool Renderer::shouldClose() const {
    return RendererWindow::shouldClose(state.window);
}

void Renderer::resize(int newW, int newH) {
    // GLFW-Callback ruft das; Größe und alle GL/CUDA-Ressourcen werden im State gehandhabt
    state.resize(newW, newH);
}

void Renderer::renderFrame() {
    FramePipeline::execute(state);
}
