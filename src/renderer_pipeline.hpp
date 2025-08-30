// Datei: src/renderer_pipeline.hpp
// 🐭 Maus-Kommentar: Die Altlast render() wurde entfernt. Nur noch drawFullscreenQuad(tex)!
// 🦦 Otter: Keine Doppelpipeline – drawFullscreenQuad ist die einzige Schnittstelle
// 🦊 Schneefuchs: „Weniger ist mehr, wenn das Mehr nur Unsinn war.“
// Struktur klar, Zweck klar, Header synchron zur Source.

#pragma once
#include <GL/glew.h> // Schneefuchs: nur was für GLuint nötig – kein PCH im Header.

namespace RendererPipeline {

// 🧱 Initialisiert Shader, VBO, VAO – Vorbereitung für Fullscreen-Quad
void init();

// 🧽 Gibt alle OpenGL-Ressourcen wieder frei
void cleanup();

// 🔁 Überträgt CUDA-PBO-Daten auf OpenGL-Textur (ohne Zeichnen)
// Muss vor drawFullscreenQuad aufgerufen werden!
void updateTexture(GLuint pbo, GLuint tex, int width, int height);

// 🎥 Zeichnet die im Texturhandle gespeicherte OpenGL-Textur fullscreen auf das Fenster
void drawFullscreenQuad(GLuint tex);

} // namespace RendererPipeline
