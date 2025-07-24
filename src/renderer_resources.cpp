// Datei: src/renderer_resources.cpp
// 🐭 Maus-Kommentar: Kontextsensitives Logging – Debug-Ausgabe nur noch bei aktiviertem Settings::debugLogging. Schneefuchs: „Finde den Ursprung, finde den Fehler.“ Keine Tippfehler mehr, keine Noise-Leaks.
#include "pch.hpp"
#include "renderer_resources.hpp"
#include "settings.hpp"
#include <stdexcept>
#include <cstdio>

namespace OpenGLUtils {

// 🕵️ Kontext-String für Logging (z.B. "init", "resize")
static const char* resourceContext = "unknown";

// Kontext setzen für folgende Ressourcenoperationen
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

    if (Settings::debugLogging)
        std::printf("[DEBUG] OpenGLUtils::createPBO -> ID %u (ctx: %s, %dx%d)\n", pbo, resourceContext, width, height);
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

    if (Settings::debugLogging)
        std::printf("[DEBUG] OpenGLUtils::createTexture -> ID %u (ctx: %s, %dx%d)\n", tex, resourceContext, width, height);
    return tex;
}

} // namespace OpenGLUtils

