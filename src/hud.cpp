// Datei: src/hud.cpp

#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"
#include "hud.hpp"
#include <vector>

#ifndef __CUDACC__
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#endif

#include <cstdio>
#include <string>

namespace Hud {

static GLuint hudVAO = 0;
static GLuint hudVBO = 0;
static GLuint hudProgram = 0;

static const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
uniform vec2 uResolution;
void main() {
    vec2 pos = aPos / uResolution * 2.0 - 1.0;
    gl_Position = vec4(pos.x, -pos.y, 0.0, 1.0); // Flip Y for top-left origin
}
)GLSL";

static const char* fragmentShaderSrc = R"GLSL(
#version 430 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 1.0, 1.0, 1.0); // White text with alpha
}
)GLSL";

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        fprintf(stderr, "[SHADER ERROR] %s\n", infoLog);
    }

    return shader;
}

GLuint createHUDProgram() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

void init() {
    glGenVertexArrays(1, &hudVAO);
    glGenBuffers(1, &hudVBO);
    hudProgram = createHUDProgram();

    // 🐭 Blending für transparente Schrift
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void drawText(const std::string& text, float x, float y, float width, float height) {
    if (text.empty()) return; // 🐭 Schutz gegen leere Strings

    char buffer[99999];
    int num_quads = stb_easy_font_print(x, y, const_cast<char*>(text.c_str()), nullptr, buffer, sizeof(buffer));

    struct Vertex {
        float x, y;
    };

    std::vector<Vertex> vertices;
    vertices.reserve(num_quads * 6); // 2 Triangles pro Zeichen

    for (int i = 0; i < num_quads; ++i) {
        unsigned char* quad = reinterpret_cast<unsigned char*>(buffer) + i * 64;
        Vertex v0 = *reinterpret_cast<Vertex*>(quad +  0); // x0, y0
        Vertex v1 = *reinterpret_cast<Vertex*>(quad + 16); // x1, y1
        Vertex v2 = *reinterpret_cast<Vertex*>(quad + 32); // x2, y2
        Vertex v3 = *reinterpret_cast<Vertex*>(quad + 48); // x3, y3

        vertices.push_back(v0);
        vertices.push_back(v1);
        vertices.push_back(v2);
        vertices.push_back(v0);
        vertices.push_back(v2);
        vertices.push_back(v3);
    }

    glBindVertexArray(hudVAO);
    glBindBuffer(GL_ARRAY_BUFFER, hudVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    glUseProgram(hudProgram);
    glUniform2f(glGetUniformLocation(hudProgram, "uResolution"), width, height);

    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));

    glUseProgram(0);
    glDisableVertexAttribArray(0);
}

void draw(float fps, float zoom, float offsetX, float offsetY, int width, int height) {
    char hudText[256];
    std::snprintf(hudText, sizeof(hudText), "FPS: %.1f  Zoom: %.2f  Offset: (%.3f, %.3f)", fps, zoom, offsetX, offsetY);
    drawText(hudText, 10.0f, 20.0f, static_cast<float>(width), static_cast<float>(height));
}

void cleanup() {
    glDeleteVertexArrays(1, &hudVAO);
    glDeleteBuffers(1, &hudVBO);
    glDeleteProgram(hudProgram);
}

} // namespace Hud
