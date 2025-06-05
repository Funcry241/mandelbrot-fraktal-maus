// Datei: src/hud.cpp

#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"
#include "hud.hpp"

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
    FragColor = vec4(1.0); // White text
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
}

void drawText(const std::string& text, float x, float y, float width, float height) {
    char buffer[99999];
    int num_quads = stb_easy_font_print(0.0f, 0.0f, const_cast<char*>(text.c_str()), nullptr, buffer, sizeof(buffer));

    glBindVertexArray(hudVAO);
    glBindBuffer(GL_ARRAY_BUFFER, hudVBO);
    glBufferData(GL_ARRAY_BUFFER, num_quads * 4 * sizeof(float), buffer, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);

    glUseProgram(hudProgram);
    glUniform2f(glGetUniformLocation(hudProgram, "uResolution"), width, height);

    glDrawArrays(GL_QUADS, 0, num_quads * 4);

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
