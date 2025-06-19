// Datei: src/renderer_resources.hpp
// Zeilen: 42
// 🐭 Maus-Kommentar: Kapselt alle OpenGL- und CUDA-Ressourcen für das Rendering. Zuständig für PBO, Textur, Interop und Cleanup. Schneefuchs sagt: „Wenn Zustände trennen, dann sauber und zuständig.“

#pragma once

#include "pch.hpp"
#include <cuda_gl_interop.h>

class RendererResources {
public:
    // 🔗 CUDA/OpenGL Interop Resource
    cudaGraphicsResource_t cudaPboResource = nullptr;

    // 📦 OpenGL Buffer und Textur
    GLuint pbo = 0;
    GLuint tex = 0;

    // 🔧 Initialisierung der Ressourcen
    void init(int width, int height);

    // 🔄 Aktualisiere die Textur über PBO (für CUDA → OpenGL)
    void updateTexture(int width, int height);

    // 🧹 Aufräumen der GPU-Ressourcen
    void cleanup();
};
