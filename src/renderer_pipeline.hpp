///// Otter: Einzige öffentliche Schnittstelle: updateTexture + drawFullscreenQuad; keine Doppelpipeline.
///// Schneefuchs: Header/Source synchron; minimaler Include (GLuint); ASCII-only.
///// Maus: Altlast render() entfernt - Struktur klar, Zweck klar.
///// Datei: src/renderer_pipeline.hpp

#pragma once
#include <GL/glew.h> // nur für GLuint

namespace RendererPipeline {

// 🧱 Initialisiert Shader, VBO, VAO - Vorbereitung für Fullscreen-Quad
void init();

// 🧽 Gibt alle OpenGL-Ressourcen wieder frei
void cleanup();

// 🎥 Zeichnet die im Texturhandle gespeicherte OpenGL-Textur fullscreen auf das Fenster
void drawFullscreenQuad(GLuint tex);

} // namespace RendererPipeline
