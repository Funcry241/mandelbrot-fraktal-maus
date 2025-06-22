// Datei: src/renderer_resources.cpp
// Zeilen: 76
// 🐭 Maus-Kommentar: Jetzt mit kontextsensitivem Logging – jeder PBO/Texture-Aufruf meldet seine Herkunft (Init, Resize, Tilewechsel etc.). Schneefuchs: „Erkenne den Ursprung der Ressourcen – dann findest du den Fehler vor dem Fehler.“

#include "pch.hpp"
#include "renderer_resources.hpp"  // ✅ Korrektur: richtiger Header-Name
#include <stdexcept>
#include <cstdio>

namespace OpenGLUtils {

// 🕵️ Kontext für Logging – z. B. "resize", "init", "tileSizeChange"
static const char* resourceContext = "unknown";

// 🔧 Kontext setzen für nachfolgende Ressourcen-Erzeugung
void setGLResourceContext(const char* context) {
    resourceContext = context ? context : "unknown";
}

// 🧱 Pixel Buffer Object erzeugen
GLuint createPBO(int width, int height) {
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 width * height * 4,  // 4 Bytes pro Pixel (RGBA8)
                 nullptr,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    std::printf("[DEBUG] OpenGLUtils::createPBO → ID %u (ctx: %s, %dx%d)\n", pbo, resourceContext, width, height);
    return pbo;
}

// 🎨 Textur erzeugen
GLuint createTexture(int width, int height) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // ✨ Sanftes Herunterskalieren
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  // ✨ Sanftes Hochskalieren
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 width, height,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 nullptr);  // Speicher nur allokieren, noch keine Daten

    glBindTexture(GL_TEXTURE_2D, 0);

    std::printf("[DEBUG] OpenGLUtils::createTexture → ID %u (ctx: %s, %dx%d)\n", tex, resourceContext, width, height);
    return tex;
}

} // namespace OpenGLUtils
