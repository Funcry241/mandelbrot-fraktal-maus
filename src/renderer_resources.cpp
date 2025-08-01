// Otter
// Datei: src/renderer_resources.cpp
// 🐭 Maus-Kommentar: Kontextsensitives Logging - Debug-Ausgabe nur noch bei aktiviertem Settings::debugLogging.
// Schneefuchs: „Finde den Ursprung, finde den Fehler.“ Keine Tippfehler mehr, keine Noise-Leaks.

#include "pch.hpp"
#include "renderer_resources.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include <stdexcept>
#include <cstdio>
#include <GL/glew.h>
#include <GL/gl.h>

namespace OpenGLUtils {

// 🕵️ Kontext-String für Logging (z.B. "init", "resize")
static const char* resourceContext = "unknown";

void setGLResourceContext(const char* context) {
    resourceContext = context ? context : "unknown";
}

// Pixel Buffer Object erzeugen
GLuint createPBO(int width, int height) {
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGLUtils::createPBO -> ID %u (ctx: %s, %dx%d)", pbo, resourceContext, width, height);
        GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-ERROR] createPBO glGetError() = 0x%04X", err);
    }
    return pbo;
}

// OpenGL-Textur erzeugen
GLuint createTexture(int width, int height) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] OpenGLUtils::createTexture -> ID %u (ctx: %s, %dx%d)", tex, resourceContext, width, height);
        GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-ERROR] createTexture glGetError() = 0x%04X", err);
    }
    return tex;
}

// Upload des PBO-Inhalts in eine Textur
void updateTextureFromPBO(GLuint pbo, GLuint tex, int width, int height) {
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Binding PBO=%u and Texture=%u (ctx: %s)", pbo, tex, resourceContext);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[GL-UPLOAD] Calling glTexSubImage2D with dimensions %dx%d", width, height);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    GLenum err = glGetError();
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[GL-UPLOAD] glGetError after glTexSubImage2D = 0x%04X", err);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[GL-UPLOAD] Texture update complete, PBO and texture unbound");
}

} // namespace OpenGLUtils
