// Datei: src/opengl_utils.cpp
// 🐭 Maus-Kommentar: OpenGL Utility Layer – komprimiert für Maximalgeschwindigkeit und zukünftige Erweiterbarkeit

#include "opengl_utils.hpp"
#include <GL/glew.h>
#include <iostream>

GLuint gFullscreenVAO = 0;

static GLuint compile(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error: " << log << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return shader;
}

GLuint createProgramFromSource(const char* vertSrc, const char* fragSrc) {
    GLuint vertexShader = compile(GL_VERTEX_SHADER, vertSrc);
    GLuint fragmentShader = compile(GL_FRAGMENT_SHADER, fragSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Program link error: " << log << std::endl;
        std::exit(EXIT_FAILURE);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

void createFullscreenQuad(GLuint* vao, GLuint* vbo, GLuint* ebo) {
    constexpr float quadVertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };

    constexpr unsigned int quadIndices[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, vao);
    glGenBuffers(1, vbo);
    glGenBuffers(1, ebo);

    glBindVertexArray(*vao);

    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0); // Optional – Clean unbind

    gFullscreenVAO = *vao;
}

void deleteFullscreenQuad(GLuint* vao, GLuint* vbo, GLuint* ebo) {
    glDeleteBuffers(1, vbo);
    glDeleteBuffers(1, ebo);
    glDeleteVertexArrays(1, vao);
}
