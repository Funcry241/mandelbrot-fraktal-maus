// Datei: src/opengl_utils.hpp
// Zeilen: 28
// 🐭 Maus-Kommentar: Aufgeräumt – keine Altlasten mehr. Nur noch moderne Shader-/Quad-Erzeugung mit expliziter VAO-Nutzung. Schneefuchs meinte: „Globals raus, Klartext rein.“

#pragma once
#ifndef OPENGL_UTILS_HPP
#define OPENGL_UTILS_HPP

#ifdef __CUDACC__
// CUDA-Compiler mag <GL/glew.h> nicht, aber wir brauchen trotzdem GLuint
typedef unsigned int GLuint;
#else
#include <GL/glew.h>
#endif

namespace OpenGLUtils {

// 🎨 Shader-Utilities
GLuint createProgramFromSource(const char* vertexSrc, const char* fragmentSrc);

// 🖼️ Fullscreen-Quad-Utilities
void createFullscreenQuad(GLuint* outVAO, GLuint* outVBO, GLuint* outEBO);

} // namespace OpenGLUtils

#endif // OPENGL_UTILS_HPP
