// Datei: src/renderer_pipeline.hpp
// Zeilen: 22
// 🐭 Maus-Kommentar: Die Altlast `render()` wurde entfernt. `drawFullscreenQuad(tex)` ist der alleinige Weg. Schneefuchs: „Weniger ist mehr, wenn das Mehr nur Unsinn war.“

#pragma once

#include "pch.hpp"

namespace RendererPipeline {

// 🧱 Shader & Quad vorbereiten
void init();
void cleanup();

// 🖼️ Textur aktualisieren aus CUDA-PBO
void updateTexture(GLuint pbo, GLuint tex, int width, int height);

// 🎥 Fullscreen-Quad zeichnen mit gegebener Textur
void drawFullscreenQuad(GLuint tex);

} // namespace RendererPipeline
