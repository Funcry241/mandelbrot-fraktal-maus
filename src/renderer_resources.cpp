///// Otter: Immutable Texture-Storage; deterministischer Upload; DSA bevorzugt; weniger Roundtrips
///// Schneefuchs: Kein per-Frame PixelStore-Save/Restore im DSA-Pfad; klare Fehlerpfade; ASCII-Logs
///// Maus: PBO/Texture Utils – robuste Init/Upload-Reihenfolge; keine versteckten Seiteneffekte
///// Datei: src/renderer_resources.cpp

#include "pch.hpp"
#include "renderer_resources.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include <cstdio>
#include <limits>
#include <stdexcept>

namespace OpenGLUtils
{

namespace {
    // Kleines Kontextlabel für Logs (z.B. "init", "resize", "draw")
    const char* g_resourceContext = "unknown";

    GLint queryMaxTexSize() {
        GLint maxTex = 0;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
        return maxTex;
    }

    inline size_t rgba8Bytes(int w, int h) {
        return static_cast<size_t>(w) * static_cast<size_t>(h) * 4u;
    }

    // --- Loader-/Extension-Utilities (GLEW/GLAD-agnostisch) -----------------
    inline bool haveDSA() {
    #if defined(GLAD_GL_VERSION_4_5)
        return glCreateTextures != nullptr;
    #elif defined(GLEW_VERSION)
        return (GLEW_VERSION_4_5 || GLEW_ARB_direct_state_access);
    #elif defined(GL_VERSION_4_5)
        return glCreateTextures != nullptr;
    #else
        return false;
    #endif
    }

    inline bool haveInvalidateSubData() {
    #if defined(GLAD_GL_VERSION_4_3)
        return glInvalidateTexImage != nullptr;
    #elif defined(GLEW_VERSION)
        return (GLEW_VERSION_4_3 || GLEW_ARB_invalidate_subdata);
    #elif defined(GL_VERSION_4_3)
        return glInvalidateTexImage != nullptr;
    #else
        return false;
    #endif
    }

    inline bool hasKHRDebug() {
    #if defined(GL_KHR_debug)
      #if defined(GLEW_VERSION)
        return GLEW_KHR_debug != 0;
      #elif defined(GLAD_GL_H_)
        return glObjectLabel != nullptr;
      #else
        return true;
      #endif
    #else
        return false;
    #endif
    }

    // --- Safe labeling helpers (nur wenn Objekt existiert) ------------------
#if defined(GL_KHR_debug)
    inline void labelTextureSafe(GLuint tex, int w, int h) {
        if (!hasKHRDebug()) return;
        if (!tex) return;
        if (!glIsTexture(tex)) return;
        char label[64];
        std::snprintf(label, sizeof(label), "OTR_tex_%dx%d_%s", w, h, g_resourceContext);
        glObjectLabel(GL_TEXTURE, tex, -1, label);
    }
    inline void labelBufferSafe(GLuint buf, int w, int h) {
        if (!hasKHRDebug()) return;
        if (!buf) return;
        if (!glIsBuffer(buf)) return;
        char label[64];
        std::snprintf(label, sizeof(label), "OTR_pbo_%dx%d_%s", w, h, g_resourceContext);
        glObjectLabel(GL_BUFFER, buf, -1, label);
    }
#endif
} // anon ns

void setGLResourceContext(const char* context) {
    g_resourceContext = (context && *context) ? context : "unknown";
}

// -----------------------------------------------------------------------------
// Immutable 2D-Texture (RGBA8) anlegen – DSA-Fastpath wenn verfügbar
// -----------------------------------------------------------------------------
unsigned int createTexture(int width, int height) {
    if (width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createTexture invalid size (ctx=%s) w=%d h=%d",
                           g_resourceContext, width, height);
        }
        return 0u;
    }

    // Guard: MaxTextureSize
    const GLint maxTex = queryMaxTexSize();
    if (maxTex > 0 && (width > maxTex || height > maxTex)) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createTexture exceeds GL_MAX_TEXTURE_SIZE=%d (ctx=%s) w=%d h=%d",
                           maxTex, g_resourceContext, width, height);
        }
        return 0u;
    }

    GLuint tex = 0;

    if (haveDSA()) {
        // ---- DSA-Pfad: keine Bindings, keine globalen Seiteneffekte
        glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    #if defined(GL_KHR_debug)
        labelTextureSafe(tex, width, height);
    #endif
        glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(tex, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
        glTextureParameteri(tex, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
        glTextureParameteri(tex, GL_TEXTURE_BASE_LEVEL, 0);
        glTextureParameteri(tex, GL_TEXTURE_MAX_LEVEL,  0);

        glTextureStorage2D(tex, 1, GL_RGBA8, width, height);

        if constexpr (Settings::debugLogging) {
            const GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                LUCHS_LOG_HOST("[GL-TEX][ERR] DSA storage glError=0x%04X (ctx=%s)",
                               static_cast<unsigned>(err), g_resourceContext);
            } else {
                LUCHS_LOG_HOST("[GL-TEX] created (DSA) RGBA8 %dx%d id=%u (ctx=%s)",
                               width, height, tex, g_resourceContext);
            }
        }
        return static_cast<unsigned int>(tex);
    }

    // ---- Fallback: klassischer Bind-Pfad (State sichern/wiederherstellen)
    GLint prevActiveTex = 0, prevTex0 = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
#if defined(GL_KHR_debug)
    labelTextureSafe(tex, width, height);
#endif

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);

    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL,  0);

    if constexpr (Settings::debugLogging) {
        const GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            LUCHS_LOG_HOST("[GL-TEX][ERR] glTexStorage2D glError=0x%04X (ctx=%s)",
                           static_cast<unsigned>(err), g_resourceContext);
        } else {
            LUCHS_LOG_HOST("[GL-TEX] created RGBA8 %dx%d id=%u (ctx=%s)",
                           width, height, tex, g_resourceContext);
        }
    }

    // State wiederherstellen
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
    glActiveTexture(static_cast<GLenum>(prevActiveTex));

    return static_cast<unsigned int>(tex);
}

// -----------------------------------------------------------------------------
// PBO anlegen (GL_PIXEL_UNPACK_BUFFER) – Größe = width*height*4 (RGBA8)
// -----------------------------------------------------------------------------
unsigned int createPBO(int width, int height) {
    if (width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createPBO invalid size (ctx=%s) w=%d h=%d",
                           g_resourceContext, width, height);
        }
        return 0u;
    }

    const size_t bytes = rgba8Bytes(width, height);

    // Vorheriges UNPACK-Binding sichern (keine Seiteneffekte nach außen)
    GLint prevPBO = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevPBO);

    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
#if defined(GL_KHR_debug)
    labelBufferSafe(pbo, width, height);
#endif

    // Persistentes Mapping vermeiden – CUDA-Interop ist mit STREAM_DRAW am robustesten.
    glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, GL_STREAM_DRAW);

    if constexpr (Settings::debugLogging) {
        // Realgröße verifizieren (64-bit Query – große PBOs sicher)
        GLint64 realSize = 0;
        glGetBufferParameteri64v(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &realSize);
        const GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            LUCHS_LOG_HOST("[GL-PBO][ERR] glBufferData/glGetBufferParameteri64v err=0x%04X (ctx=%s)",
                           static_cast<unsigned>(err), g_resourceContext);
        }
        LUCHS_LOG_HOST("[GL-PBO] created unpack PBO=%u requested=%zu real=%lld (ctx=%s)",
                       pbo, bytes, static_cast<long long>(realSize), g_resourceContext);
    }

    // Vorheriges Binding wiederherstellen
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevPBO));

    return static_cast<unsigned int>(pbo);
}

// -----------------------------------------------------------------------------
// Texture-Update aus UNPACK-PBO (Vollflächen-Upload) – DSA bevorzugt
// -----------------------------------------------------------------------------
void updateTextureFromPBO(unsigned int pboU, unsigned int texU, int width, int height) {
    if (!pboU || !texU || width <= 0 || height <= 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] updateTextureFromPBO invalid args pbo=%u tex=%u w=%d h=%d",
                           pboU, texU, width, height);
        }
        return;
    }

    const GLuint pbo = static_cast<GLuint>(pboU);
    const GLuint tex = static_cast<GLuint>(texU);

    // Quelle: gebundener UNPACK-PBO – nur kurzzeitig binden; PixelStore
    // Policy wurde bei Resize/Init einmalig gesetzt (UNPACK_ALIGNMENT=1 etc.).
    GLint prevPBO = 0;
    glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    if (haveInvalidateSubData()) {
        // Kein Label/Bind notwendig – nur invalidieren, wenn Textur existiert
        if (glIsTexture(tex)) {
            glInvalidateTexImage(tex, 0);
        }
    }

    if (haveDSA()) {
        // DSA: kein Textur-Bind nötig; kein per-Frame PixelStore-Save/Restore
        glTextureSubImage2D(tex, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        if constexpr (Settings::debugLogging) {
            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                LUCHS_LOG_HOST("[GL-UPLOAD][ERR] DSA glTextureSubImage2D glError=0x%04X (ctx=%s)",
                               static_cast<unsigned>(err), g_resourceContext);
            } else {
                LUCHS_LOG_HOST("[GL-UPLOAD] done (DSA) ctx=%s", g_resourceContext);
            }
        }
    } else {
        // Fallback: Bind/Unbind-Pfad – minimaler State-Touch
        GLint prevActiveTex = 0, prevTex0 = 0;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
        glActiveTexture(GL_TEXTURE0);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0,
                        0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE,
                        nullptr); // Quelle: gebundener PBO

        if constexpr (Settings::debugLogging) {
            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                LUCHS_LOG_HOST("[GL-UPLOAD][ERR] glTexSubImage2D glError=0x%04X (ctx=%s)",
                               static_cast<unsigned>(err), g_resourceContext);
            } else {
                LUCHS_LOG_HOST("[GL-UPLOAD] done (bind) ctx=%s", g_resourceContext);
            }
        }

        // Textur/ActiveTexture restaurieren
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prevTex0));
        glActiveTexture(static_cast<GLenum>(prevActiveTex));
    }

    // UNPACK-PBO restaurieren
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevPBO));
}

} // namespace OpenGLUtils
