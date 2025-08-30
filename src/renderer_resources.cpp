// Datei: src/renderer_resources.cpp
///// MAUS: OpenGL PBO/Texture utils — deterministic upload & ASCII logs
///// Otter: Immutable Texture-Storage + sauberes PixelStore-Handling. Upload deterministisch.
///// Schneefuchs: State sauber sichern/wiederherstellen; eindeutige Fehlerpfade. ASCII-Logs.

#include "pch.hpp"
#include "renderer_resources.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include <stdexcept>
#include <limits>
#include <GL/glew.h>

namespace OpenGLUtils
{

// Context label for logs (e.g., "init", "resize")
static const char* resourceContext = "unknown";

void setGLResourceContext(const char* context) {
    resourceContext = context ? context : "unknown";
}

// Create Pixel Buffer Object
GLuint createPBO(int width, int height) {
    if (width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createPBO invalid size (ctx=%s) w=%d h=%d", resourceContext, width, height);
        }
        return 0;
    }

    const unsigned long long w = static_cast<unsigned long long>(width);
    const unsigned long long h = static_cast<unsigned long long>(height);
    const unsigned long long total = w * h * 4ull; // RGBA8
    if (total > static_cast<unsigned long long>(std::numeric_limits<GLsizeiptr>::max())) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createPBO size overflow (ctx=%s) w=%llu h=%llu bytes=%llu",
                           resourceContext, w, h, total);
        }
        return 0;
    }
    const GLsizeiptr bytes = static_cast<GLsizeiptr>(total);

    GLint prevPBO = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevPBO);

    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    if (!pbo) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] glGenBuffers failed for PBO (ctx=%s)", resourceContext);
        }
        return 0;
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, nullptr, GL_STREAM_DRAW); // orphan on (re)alloc
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevPBO));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGLUtils::createPBO -> ID %u (ctx=%s, %dx%d, bytes=%lld)",
                       pbo, resourceContext, width, height, static_cast<long long>(bytes));
        const GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-ERROR] createPBO glGetError() = 0x%04X", err);
    }
    return pbo;
}

// Create immutable 2D texture (RGBA8)
GLuint createTexture(int width, int height) {
    if (width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createTexture invalid size (ctx=%s) w=%d h=%d", resourceContext, width, height);
        }
        return 0;
    }

    GLint prevActiveTex = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
    GLint prevTex2D = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex2D);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    if (!tex) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] glGenTextures failed (ctx=%s)", resourceContext);
        }
        return 0;
    }

    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);

    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);

    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex2D));
    glActiveTexture(static_cast<GLenum>(prevActiveTex));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGLUtils::createTexture -> ID %u (ctx=%s, %dx%d, RGBA8 immutable)",
                       tex, resourceContext, width, height);
        const GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-ERROR] createTexture glGetError() = 0x%04X", err);
    }
    return tex;
}

// Upload PBO -> texture
void updateTextureFromPBO(GLuint pbo, GLuint tex, int width, int height) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Binding PBO=%u and Texture=%u (ctx=%s, %dx%d)",
                       pbo, tex, resourceContext, width, height);
    }

    GLint prevActiveTex = 0, prevTex2D = 0, prevPBO = 0;
    GLint prevAlign = 0, prevRowLen = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex2D);
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevPBO);
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevAlign);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &prevRowLen);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    if constexpr (Settings::debugLogging) {
        const GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-UPLOAD] glTexSubImage2D glGetError() = 0x%04X", err);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevPBO));
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex2D));
    glPixelStorei(GL_UNPACK_ALIGNMENT, prevAlign);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, prevRowLen);
    glActiveTexture(static_cast<GLenum>(prevActiveTex));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Texture update complete, state restored");
    }
}

} // namespace OpenGLUtils

// -----------------------------------------------------------------------------
// Back-compat adapters (global namespace) — keep older call sites stable.
// -----------------------------------------------------------------------------
namespace {
    // Cached last resource IDs for the 0-arg context setter
    GLuint g_cachedTextureId = 0;
    GLuint g_cachedPboId     = 0;
}

void setGLResourceContext() noexcept {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL] setGLResourceContext (compat 0-arg): texture=%u, pbo=%u",
                       static_cast<unsigned>(g_cachedTextureId),
                       static_cast<unsigned>(g_cachedPboId));
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_cachedPboId);
    glBindTexture(GL_TEXTURE_2D, g_cachedTextureId);
}

void setGLResourceContext(GLuint textureId, GLuint pboId) noexcept {
    g_cachedTextureId = textureId;
    g_cachedPboId     = pboId;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL] setGLResourceContext (compat 2-arg): texture=%u, pbo=%u",
                       static_cast<unsigned>(textureId),
                       static_cast<unsigned>(pboId));
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
    glBindTexture(GL_TEXTURE_2D, textureId);
}

void updateTextureFromPBO(GLuint textureId, GLsizei width, GLsizei height) noexcept {
    GLint boundPbo = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundPbo);
    if (boundPbo == 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[GL][ERR] compat updateTextureFromPBO: no PBO bound; texture=%u, %dx%d",
                           static_cast<unsigned>(textureId),
                           static_cast<int>(width),
                           static_cast<int>(height));
        }
        return;
    }

    OpenGLUtils::updateTextureFromPBO(static_cast<GLuint>(boundPbo),
                                      textureId,
                                      static_cast<int>(width),
                                      static_cast<int>(height));
}

void updateTextureFromPBO(GLuint textureId, GLuint pboId, int width, int height) noexcept {
    // Observed today: call site passes (texture, pbo, w, h)
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL] compat updateTextureFromPBO(4): texture=%u, pbo=%u, %dx%d",
                       static_cast<unsigned>(textureId),
                       static_cast<unsigned>(pboId),
                       width, height);
    }
    OpenGLUtils::updateTextureFromPBO(pboId, textureId, width, height);
}
