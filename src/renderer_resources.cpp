// Datei: src/renderer_resources.cpp
// Zeilen: 74
// 🐭 Maus-Kommentar: Verwaltet PBO, Textur und CUDA-Interop. Verwendet robuste `OpenGLUtils`-Hilfen. `init()` und `cleanup()` sind strikt symmetrisch. Schneefuchs: „Wer allokiert, der räumt auch auf.“

#include "pch.hpp"
#include "renderer_resources.hpp"
#include "opengl_utils.hpp"
#include "common.hpp"

void RendererResources::init(int width, int height) {
    // 🔁 Vorherige Ressourcen ggf. freigeben
    cleanup();

    // 🧵 OpenGL: PBO + Textur erstellen
    pbo = OpenGLUtils::createPBO(width, height);
    tex = OpenGLUtils::createTexture(width, height);

    // 🔗 CUDA-Interop registrieren
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

#if defined(DEBUG) || defined(_DEBUG)
    std::printf("[DEBUG] RendererResources::init → PBO %u, Texture %u\n", pbo, tex);
#endif
}

void RendererResources::updateTexture(int width, int height) {
    // 🔁 Texturinhalt vom PBO übernehmen (CUDA → OpenGL)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexSubImage2D(GL_TEXTURE_2D,
                    0, 0, 0,
                    width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE,
                    nullptr);  // Daten liegen im gebundenen PBO

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void RendererResources::cleanup() {
    // 🔌 CUDA-Interop deregistrieren
    if (cudaPboResource) {
        cudaGraphicsUnregisterResource(cudaPboResource);
        cudaPboResource = nullptr;
    }

    // 🧹 OpenGL-Objekte löschen
    if (pbo) {
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }

    if (tex) {
        glDeleteTextures(1, &tex);
        tex = 0;
    }
}
