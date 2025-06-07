#pragma once

#include <GL/glew.h>

extern GLuint gFullscreenVAO;

GLuint createProgramFromSource(const char* vertSrc, const char* fragSrc);
void createFullscreenQuad(GLuint* vao, GLuint* vbo, GLuint* ebo);
void drawFullscreenQuad();
void deleteFullscreenQuad(GLuint* vao, GLuint* vbo, GLuint* ebo);

