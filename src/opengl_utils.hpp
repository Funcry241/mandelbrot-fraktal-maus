// Datei: src/opengl_utils.hpp
// Zeilen: 35
// 🐭 Maus-Kommentar: Header für OpenGL-Hilfsfunktionen – VAO für Fullscreen-Rendering, Shader-Erzeugung aus Quelltext. GLEW wird nur eingebunden, wenn **nicht** im CUDA-Compiler, sonst gibt es Symbolkonflikte. Schneefuchs hätte den CUDA-Ausschluss beim VAO geliebt – sonst kracht's bei `nvcc`.

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

// 🖥️ Globale VAO-ID für das Fullscreen-Quad
#ifndef __CUDACC__
extern GLuint gFullscreenVAO;
#endif

// 🎨 Shader-Utilities
GLuint createProgramFromSource(const char* vertexSrc, const char* fragmentSrc);

// 🖼️ Fullscreen-Quad-Utilities
void createFullscreenQuad(GLuint* outVAO, GLuint* outVBO, GLuint* outEBO);
void drawFullscreenQuad();
void deleteFullscreenQuad(GLuint* inVAO, GLuint* inVBO, GLuint* inEBO);

} // namespace OpenGLUtils

#endif // OPENGL_UTILS_HPP
