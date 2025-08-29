///// MAUS: OpenGL PBO/Texture utils — deterministic upload & ASCII logs
// Datei: src/renderer_resources.cpp
// 🐭 Maus-Kommentar: Kontextsensitives Logging – Debug nur bei Settings::debugLogging.
// 🦦 Otter: Immutable Texture-Storage + sauberes PixelStore-Handling. Upload deterministisch. (Bezug zu Otter)
// 🦊 Schneefuchs: State sauber sichern/wiederherstellen; eindeutige Fehlerpfade. ASCII-Logs. (Bezug zu Schneefuchs)

#include "pch.hpp"
#include "renderer_resources.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include <stdexcept>
#include <cstdio>
#include <GL/glew.h>
#include <GL/gl.h>

namespace OpenGLUtils
{

// 🕵️ Kontext-String für Logging (z.B. "init", "resize")
static const char* resourceContext = "unknown";

void setGLResourceContext(const char* context) {
    resourceContext = context ? context : "unknown";
}

// Pixel Buffer Object erzeugen
GLuint createPBO(int width, int height) {
    if (width <= 0 || height <= 0) {
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createPBO invalid size (ctx=%s) w=%d h=%d", resourceContext, width, height);
        }
        return 0;
    }

    const GLsizeiptr bytes = static_cast<GLsizeiptr>(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);

    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, nullptr, GL_STREAM_DRAW); // orphan on (re)alloc
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGLUtils::createPBO -> ID %u (ctx=%s, %dx%d, bytes=%lld)",
                       pbo, resourceContext, width, height, static_cast<long long>(bytes));
        GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-ERROR] createPBO glGetError() = 0x%04X", err);
    }
    return pbo;
}

// OpenGL-Textur erzeugen (immutable storage, dann SubImage-Updates)
GLuint createTexture(int width, int height) {
    if (width <= 0 || height <= 0) {
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ERROR] createTexture invalid size (ctx=%s) w=%d h=%d", resourceContext, width, height);
        }
        return 0;
    }

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    // Parameter einmalig setzen
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);

    // 🦦 Otter: Immutable storage – sauberer als TexImage2D, kompatibel zu SubImage2D
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);

    glBindTexture(GL_TEXTURE_2D, 0);

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGLUtils::createTexture -> ID %u (ctx=%s, %dx%d, RGBA8 immutable)",
                       tex, resourceContext, width, height);
        GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-ERROR] createTexture glGetError() = 0x%04X", err);
    }
    return tex;
}

// Upload des PBO-Inhalts in eine Textur
void updateTextureFromPBO(GLuint pbo, GLuint tex, int width, int height) {
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Binding PBO=%u and Texture=%u (ctx=%s, %dx%d)",
                       pbo, tex, resourceContext, width, height);
    }

    // 🐑 Schneefuchs: Vorherige PixelStore-Werte sichern, deterministisch setzen, am Ende restaurieren.
    GLint prevAlign = 0, prevRowLen = 0;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevAlign);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &prevRowLen);

    // **Wichtig**: Einheit wählen, dann Textur binden (robust gegen Fremdstate)
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // sicher für beliebige Breiten
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE, nullptr); // liest aus PBO offset 0

    GLenum err = glGetError();
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] glTexSubImage2D glGetError() = 0x%04X", err);
    }

    // State restaurieren
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, prevAlign);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, prevRowLen);

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Texture update complete, state restored");
    }
}

} // namespace OpenGLUtils
