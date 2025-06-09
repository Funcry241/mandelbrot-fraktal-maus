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
