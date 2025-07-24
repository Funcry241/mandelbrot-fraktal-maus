// Datei: src/renderer_pipeline.hpp
// 🐭 Maus-Kommentar: Die Altlast render() wurde entfernt. Nur noch drawFullscreenQuad(tex)! Schneefuchs: „Weniger ist mehr, wenn das Mehr nur Unsinn war.“

#pragma once
#include "pch.hpp"

namespace RendererPipeline {

// 🧱 Shader & Quad vorbereiten
void init();
void cleanup();

// 🖼️ CUDA-PBO auf OpenGL-Textur aktualisieren
void updateTexture(GLuint pbo, GLuint tex, int width, int height);

// 🎥 Fullscreen-Quad zeichnen (Textur)
void drawFullscreenQuad(GLuint tex);

} // namespace RendererPipeline
