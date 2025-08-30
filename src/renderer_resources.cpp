///// Otter: Immutable Texture-Storage; deterministischer Upload; sauberes PixelStore-Handling.
///// Schneefuchs: State sichern/wiederherstellen; klare Fehlerpfade; ASCII-Logs.
///// Maus: OpenGL PBO/Texture Utils – robuste Init/Upload-Reihenfolge; keine versteckten Seiteneffekte.

#include "pch.hpp"
#include "renderer_resources.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include <stdexcept>
#include <limits>

namespace OpenGLUtils
{

// Context label for logs (e.g., "init", "resize")
static const char* resourceContext = "unknown";

void setGLResourceContext(const char* context) {
    resourceContext = context ? context : "unknown";
}

// Create Pixel Buffer Object (for GL_PIXEL_UNPACK_BUFFER uploads)
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
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, nullptr, GL_STREAM_DRAW); // upload path

    // verify real size
    GLint realSize = 0;
    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &realSize);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevPBO));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGLUtils::createPBO -> ID %u (ctx=%s, %dx%d, bytes=%lld real=%d)",
                       pbo, resourceContext, width, height, static_cast<long long>(bytes), realSize);
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

    // Work on unit 0 to avoid surprises; save its binding
    glActiveTexture(GL_TEXTURE0);
    GLint prevTex0 = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    if (!tex) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] glGenTextures failed (ctx=%s)", resourceContext);
        }
        // restore active unit
        glActiveTexture(static_cast<GLenum>(prevActiveTex));
        return 0;
    }

    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);

    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);

    // restore unit 0 binding and previous active unit
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
    glActiveTexture(static_cast<GLenum>(prevActiveTex));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGLUtils::createTexture -> ID %u (ctx=%s, %dx%d, RGBA8 immutable)",
                       tex, resourceContext, width, height);
        const GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-ERROR] createTexture glGetError() = 0x%04X", err);
    }
    return tex;
}

// Upload PBO -> texture (RGBA8), preserving global state
void updateTextureFromPBO(GLuint pbo, GLuint tex, int width, int height) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Binding PBO=%u and Texture=%u (ctx=%s, %dx%d)",
                       pbo, tex, resourceContext, width, height);
    }

    if (pbo == 0 || tex == 0 || width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[GL-UPLOAD][ERR] invalid args pbo=%u tex=%u w=%d h=%d", pbo, tex, width, height);
        }
        return;
    }

    // Save global state
    GLint prevActiveTex = 0, prevPBO = 0, prevAlign = 0, prevRowLen = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevPBO);
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevAlign);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &prevRowLen);

    // Switch to unit 0 (Sampler 0) and save its 2D binding
    glActiveTexture(GL_TEXTURE0);
    GLint prevTex0 = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

    // Upload from PBO -> tex (uchar4 layout)
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    glTexSubImage2D(GL_TEXTURE_2D, 0,
                    0, 0, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE,
                    nullptr); // from bound PBO

    if constexpr (Settings::debugLogging) {
        const GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-UPLOAD] glTexSubImage2D glGetError() = 0x%04X", err);
    }

    // Restore unit 0 binding, then global state (order matters!)
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevPBO));
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
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL] compat updateTextureFromPBO(4): texture=%u, pbo=%u, %dx%d",
                       static_cast<unsigned>(textureId),
                       static_cast<unsigned>(pboId),
                       width, height);
    }
    OpenGLUtils::updateTextureFromPBO(pboId, textureId, width, height);
}
