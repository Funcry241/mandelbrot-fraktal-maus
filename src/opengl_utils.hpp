// Datei: src/opengl_utils.hpp

#pragma once

#include <GL/glew.h>

// Globale VAO-ID verfügbar machen
extern GLuint gFullscreenVAO;

// Shader-Utils
GLuint createProgramFromSource(const char* vertexSrc, const char* fragmentSrc);

// Fullscreen-Quad-Utils
void createFullscreenQuad(GLuint* outVAO, GLuint* outVBO, GLuint* outEBO);
void drawFullscreenQuad();
void deleteFullscreenQuad(GLuint* inVAO, GLuint* inVBO, GLuint* inEBO);