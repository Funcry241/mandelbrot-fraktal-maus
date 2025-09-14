///// Otter: Immutable Texture-Storage; deterministischer Upload; sauberes PixelStore-Handling.
///// Schneefuchs: State sichern/wiederherstellen; klare Fehlerpfade; ASCII-Logs.
///// Maus: OpenGL PBO/Texture Utils – robuste Init/Upload-Reihenfolge; keine versteckten Seiteneffekte.
///// Datei: src/renderer_resources.cpp

#include "pch.hpp"
#include "renderer_resources.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include <stdexcept>
#include <limits>

namespace OpenGLUtils
{

// -----------------------------------------------------------------------------
// TU-lokaler Kontext-String für Logs (z. B. "init", "resize")
// -----------------------------------------------------------------------------
static const char* g_resourceContext = "unknown";

void setGLResourceContext(const char* context) {
    g_resourceContext = context ? context : "unknown";
}

// Kleiner Helper: Maximal erlaubte Texture-Größe (für Diagnose/Guards)
static inline GLint queryMaxTexSize() {
    GLint maxTex = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
    return maxTex;
}

// -----------------------------------------------------------------------------
// Immutable 2D-Texture (RGBA8) anlegen
// -----------------------------------------------------------------------------
GLuint createTexture(int width, int height) {
    if (width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createTexture invalid size (ctx=%s) w=%d h=%d",
                           g_resourceContext, width, height);
        }
        return 0;
    }

    // Guard: MaxTextureSize
    const GLint maxTex = queryMaxTexSize();
    if (maxTex > 0 && (width > maxTex || height > maxTex)) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createTexture exceeds GL_MAX_TEXTURE_SIZE=%d (ctx=%s) w=%d h=%d",
                           maxTex, g_resourceContext, width, height);
        }
        return 0;
    }

    // Globalen State sichern
    GLint prevActiveTex = 0, prevTex0 = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    // Sampler-Parameter (stabil, keine Mipmaps)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);

    // Immutable Storage
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);

    // Clamp to single mip level (no mips)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL,  0);

    // Bindings/Einstellungen wiederherstellen
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
    glActiveTexture(static_cast<GLenum>(prevActiveTex));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-TEX] created RGBA8 %dx%d id=%u (ctx=%s)",
                       width, height, tex, g_resourceContext);
    }
    return tex;
}

// -----------------------------------------------------------------------------
// PBO anlegen (GL_PIXEL_UNPACK_BUFFER) – Größe = width*height*4 (RGBA8)
// -----------------------------------------------------------------------------
GLuint createPBO(int width, int height) {
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-PBO] created unpack PBO=%u bytes=%zu (ctx=%s)", pbo, bytes, g_resourceContext);
    }
    return pbo;
}

// -----------------------------------------------------------------------------
// Texture-Update aus gebundenem UNPACK-PBO (Vollflächen-Upload)
// -----------------------------------------------------------------------------
void updateTextureFromPBO(GLuint pbo, GLuint tex, int width, int height) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Binding PBO=%u and Texture=%u (ctx=%s, %dx%d)",
                       pbo, tex, g_resourceContext, width, height);
    }

    if (pbo == 0 || tex == 0 || width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[GL-UPLOAD][ERR] invalid args pbo=%u tex=%u w=%d h=%d",
                           pbo, tex, width, height);
        }
        return;
    }

    // Globalen State sichern
    GLint prevActiveTex = 0, prevPBO = 0, prevAlign = 0, prevRowLen = 0;
    GLint prevSkipPix = 0, prevSkipRows = 0, prevTex0 = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevPBO);
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevAlign);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &prevRowLen);
    glGetIntegerv(GL_UNPACK_SKIP_PIXELS, &prevSkipPix);
    glGetIntegerv(GL_UNPACK_SKIP_ROWS, &prevSkipRows);

    // Set: Bind Ziel-Texture & PBO
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // PixelStore-Einstellungen
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS,   0);

    glInvalidateTexImage(tex, 0);

    glTexSubImage2D(GL_TEXTURE_2D, 0,
                    0, 0, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE,
                    nullptr); // Quelle: gebundener PBO

    if constexpr (Settings::debugLogging) {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            LUCHS_LOG_HOST("[GL-UPLOAD][ERR] glTexSubImage2D glGetError()=0x%04X", (unsigned)err);
        }
    }

    // State wiederherstellen
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevPBO));
    glPixelStorei(GL_UNPACK_ALIGNMENT, prevAlign);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, prevRowLen);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, prevSkipPix);
    glPixelStorei(GL_UNPACK_SKIP_ROWS,   prevSkipRows);
    glActiveTexture(static_cast<GLenum>(prevActiveTex));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] done (ctx=%s)", g_resourceContext);
    }
}

} // namespace OpenGLUtils
