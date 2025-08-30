// 🐭 Maus-Kommentar: Robbe-Edition - keine Altlasten mehr. Moderne Shader-/Quad-Erzeugung, explizite VAO-Nutzung. GLEW/GLFW IMMER im PCH. Schneefuchs: „Header bleibt schlank, sonst beißt die Robbe!“

#pragma once
#ifndef OPENGL_UTILS_HPP
#define OPENGL_UTILS_HPP

#include <GL/glew.h> // Schneefuchs: Nur was für GLuint nötig ist; kein PCH im Header.

namespace OpenGLUtils {

// 🎨 Shader-Utilities
[[nodiscard]] GLuint createProgramFromSource(const char* vertexSrc, const char* fragmentSrc);

// 🖼️ Fullscreen-Quad-Utilities
void createFullscreenQuad(GLuint* outVAO, GLuint* outVBO, GLuint* outEBO);

} // namespace OpenGLUtils

#endif // OPENGL_UTILS_HPP
