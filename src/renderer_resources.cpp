// Datei: src/opengl_utils.cpp
// Zeilen: 55
// 🐭 Maus-Kommentar: Implementiert Hilfsfunktionen zur Erstellung von OpenGL-PBOs und Texturen für CUDA-Interop. Korrekt initialisierte Objekte vermeiden undefined behavior. Schneefuchs: „Kein Fraktal ohne Puffer – und kein Puffer ohne Format!“

#include "pch.hpp"
#include "opengl_utils.hpp"
#include <stdexcept>
#include <cstdio>

namespace OpenGLUtils {

GLuint createPBO(int width, int height) {
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 width * height * 4,  // 4 Bytes pro Pixel (RGBA8)
                 nullptr,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

#if defined(DEBUG) || defined(_DEBUG)
    std::printf("[DEBUG] OpenGLUtils::createPBO → ID %u\n", pbo);
#endif

    return pbo;
}

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

#if defined(DEBUG) || defined(_DEBUG)
    std::printf("[DEBUG] OpenGLUtils::createTexture → ID %u\n", tex);
#endif

    return tex;
}

} // namespace OpenGLUtils
