// Datei: src/opengl_utils.hpp
// Zeilen: 24
// 🐭 Maus-Kommentar: Robbe-Edition – keine Altlasten mehr. Nur noch moderne Shader-/Quad-Erzeugung mit expliziter VAO-Nutzung. GLEW/GLFW jetzt IMMER zentral im PCH. Schneefuchs: „Header bleibt schlank, sonst beißt die Robbe!“

#pragma once
#ifndef OPENGL_UTILS_HPP
#define OPENGL_UTILS_HPP

#include "pch.hpp" // Enthält GLEW, GLFW, CUDA usw.

namespace OpenGLUtils {

// 🎨 Shader-Utilities
GLuint createProgramFromSource(const char* vertexSrc, const char* fragmentSrc);

// 🖼️ Fullscreen-Quad-Utilities
void createFullscreenQuad(GLuint* outVAO, GLuint* outVBO, GLuint* outEBO);

} // namespace OpenGLUtils

#endif // OPENGL_UTILS_HPP
