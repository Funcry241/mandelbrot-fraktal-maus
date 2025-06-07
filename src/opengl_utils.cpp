// Datei: src/opengl_utils.cpp
// 🐭 Maus-Kommentar: OpenGL Utility Layer – komprimiert auf Maximalgeschwindigkeit

#include "opengl_utils.hpp"
#include <GL/glew.h>
#include <iostream>

GLuint gFullscreenVAO = 0;

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader compile error: " << log << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return s;
}

GLuint createProgramFromSource(const char* vertSrc, const char* fragSrc) {
    GLuint v = compile(GL_VERTEX_SHADER, vertSrc), f = compile(GL_FRAGMENT_SHADER, fragSrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, 512, nullptr, log);
        std::cerr << "Program link error: " << log << std::endl;
        std::exit(EXIT_FAILURE);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

void createFullscreenQuad(GLuint* vao, GLuint* vbo, GLuint* ebo) {
    float quad[] = { -1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 0, 1 };
    unsigned idx[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, vao);
    glGenBuffers(1, vbo);
    glGenBuffers(1, ebo);

    glBindVertexArray(*vao);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    gFullscreenVAO = *vao;
}

void drawFullscreenQuad() {
    glBindVertexArray(gFullscreenVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void deleteFullscreenQuad(GLuint* vao, GLuint* vbo, GLuint* ebo) {
    glDeleteBuffers(1, vbo);
    glDeleteBuffers(1, ebo);
    glDeleteVertexArrays(1, vao);
}
