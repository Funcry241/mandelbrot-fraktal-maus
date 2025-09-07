///// Otter: OpenGL-Utils (Header) – schlank; nur GLuint-Typ, klare API.
///// Schneefuchs: Header/Source synchron; deterministisch; ASCII-only; keine Seiteneffekte.
///// Maus: GLEW/GLFW im PCH ok; hier minimal halten – sonst beißt die Robbe!
///// Datei: src/opengl_utils.hpp

#pragma once
#ifndef OPENGL_UTILS_HPP
#define OPENGL_UTILS_HPP

#include <GL/glew.h> // Für GLuint (leichtgewichtig genug; sonst via PCH eingebunden)

namespace OpenGLUtils {

// 🎨 Shader-Utilities
// Erzeugt ein GL-Program aus Vertex/Fragment-Quelltexten.
// Rückgabe: Program-ID (0 bei Fehler). Wirft nicht.
[[nodiscard]] GLuint createProgramFromSource(const char* vertexSrc,
                                             const char* fragmentSrc) noexcept;

// 🖼️ Fullscreen-Quad-Utilities
// Erstellt einen einfachen FSQ (VAO/VBO/EBO). Existierende Ziele werden überschrieben.
// Alle Pointer müssen gültig sein. Wirft nicht.
void createFullscreenQuad(GLuint* outVAO,
                          GLuint* outVBO,
                          GLuint* outEBO) noexcept;

} // namespace OpenGLUtils

#endif // OPENGL_UTILS_HPP
