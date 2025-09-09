///// Otter: Immutable Texture-Storage; deterministischer Upload; sauberes PixelStore-Handling.
///// Schneefuchs: State sichern/wiederherstellen; klare Fehlerpfade; ASCII-Logs.
///// Maus: OpenGL PBO/Texture Utils – robuste Init/Upload-Reihenfolge; keine versteckten Seiteneffekte.
//  CUDA 13 Kontext: Host-seitig; keine Device-Abhängigkeiten. Logs & Checks minimal im Fastpath.
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
    return (maxTex > 0) ? maxTex : 0;
}

// -----------------------------------------------------------------------------
// PBO anlegen (GL_PIXEL_UNPACK_BUFFER) – RGBA8 Uploadpfad
// -----------------------------------------------------------------------------
GLuint createPBO(int width, int height) {
    if (width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createPBO invalid size (ctx=%s) w=%d h=%d",
                           g_resourceContext, width, height);
        }
        return 0;
    }

    const unsigned long long w = static_cast<unsigned long long>(width);
    const unsigned long long h = static_cast<unsigned long long>(height);
    const unsigned long long total = w * h * 4ull; // RGBA8
    if (total > static_cast<unsigned long long>(std::numeric_limits<GLsizeiptr>::max())) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createPBO size overflow (ctx=%s) w=%llu h=%llu bytes=%llu",
                           g_resourceContext, w, h, total);
        }
        return 0;
    }
    const GLsizeiptr bytes = static_cast<GLsizeiptr>(total);

    // Vorheriges Binding sichern
    GLint prevPBO = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevPBO);

    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    if (!pbo) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] glGenBuffers failed for PBO (ctx=%s)", g_resourceContext);
        }
        return 0;
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, nullptr, GL_STREAM_DRAW); // Upload-Pfad

    // Realgröße verifizieren (Debug)
    GLint realSize = 0;
    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &realSize);

    // Binding wiederherstellen
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevPBO));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGLUtils::createPBO -> ID %u (ctx=%s, %dx%d, bytes=%lld real=%d)",
                       pbo, g_resourceContext, width, height,
                       static_cast<long long>(bytes), realSize);
        const GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-ERROR] createPBO glGetError() = 0x%04X", err);
    }
    return pbo;
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

    // Aktive Texture Einheit sichern
    GLint prevActiveTex = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);

    // Auf Unit 0 arbeiten; deren Binding sichern
    glActiveTexture(GL_TEXTURE0);
    GLint prevTex0 = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    if (!tex) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] glGenTextures failed (ctx=%s)", g_resourceContext);
        }
        // Vorherige aktive Einheit wiederherstellen
        glActiveTexture(static_cast<GLenum>(prevActiveTex));
        return 0;
    }

    glBindTexture(GL_TEXTURE_2D, tex);
    // Sampler-Parameter (stabil, keine Mipmaps)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);

    // Immutable Storage
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);

    // Bindings/Einstellungen wiederherstellen
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
    glActiveTexture(static_cast<GLenum>(prevActiveTex));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGLUtils::createTexture -> ID %u (ctx=%s, %dx%d, RGBA8 immutable)",
                       tex, g_resourceContext, width, height);
        const GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-ERROR] createTexture glGetError() = 0x%04X", err);
    }
    return tex;
}

// -----------------------------------------------------------------------------
// Upload PBO -> Texture (RGBA8), globalen GL-State vollständig restaurieren
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
    GLint prevSkipPix = 0, prevSkipRows = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevPBO);
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevAlign);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &prevRowLen);
    glGetIntegerv(GL_UNPACK_SKIP_PIXELS, &prevSkipPix);
    glGetIntegerv(GL_UNPACK_SKIP_ROWS,   &prevSkipRows);

    // Auf Unit 0 umschalten; deren 2D-Binding sichern
    glActiveTexture(GL_TEXTURE0);
    GLint prevTex0 = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

    // Upload aus PBO → Texture (uchar4-Layout)
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Deterministische PixelStore-Einstellungen
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS,   0);

    glTexSubImage2D(GL_TEXTURE_2D, 0,
                    0, 0, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE,
                    nullptr); // Quelle: gebundener PBO

    if constexpr (Settings::debugLogging) {
        const GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-UPLOAD] glTexSubImage2D glGetError() = 0x%04X", err);
    }

    // Reihenfolge der Wiederherstellung beachten!
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevPBO));
    glPixelStorei(GL_UNPACK_ALIGNMENT, prevAlign);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, prevRowLen);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, prevSkipPix);
    glPixelStorei(GL_UNPACK_SKIP_ROWS,   prevSkipRows);
    glActiveTexture(static_cast<GLenum>(prevActiveTex));

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Texture update complete, state restored");
    }
}

} // namespace OpenGLUtils
