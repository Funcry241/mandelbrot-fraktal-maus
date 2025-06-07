// Datei: src/hud.cpp
// 🐭 Maus-Kommentar: Ultrakompakter HUD-Renderer mit minimalem Shader-Overhead

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

static GLuint vao = 0, vbo = 0, prog = 0;

static constexpr const char* vertSrc = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;
uniform vec2 uResolution;
void main() { vec2 p = aPos / uResolution * 2.0 - 1.0; gl_Position = vec4(p.x, -p.y, 0.0, 1.0); }
)GLSL";

static constexpr const char* fragSrc = R"GLSL(
#version 430 core
out vec4 FragColor;
void main() { FragColor = vec4(1.0); }
)GLSL";

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        fprintf(stderr, "[SHADER ERROR] %s\n", log);
    }
    return s;
}

GLuint createProgram() {
    GLuint vs = compile(GL_VERTEX_SHADER, vertSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fragSrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

void init() {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    prog = createProgram();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void drawText(const std::string& text, float x, float y, float w, float h) {
    if (text.empty()) return;
    char buf[99999];
    int quads = stb_easy_font_print(x, y, const_cast<char*>(text.c_str()), nullptr, buf, sizeof(buf));
    struct V { float x, y; };
    std::vector<V> verts;
    verts.reserve(quads * 6);
    for (int i = 0; i < quads; ++i) {
        auto* q = reinterpret_cast<V*>(buf + i * 64);
        verts.push_back(q[0]); verts.push_back(q[1]); verts.push_back(q[2]);
        verts.push_back(q[0]); verts.push_back(q[2]); verts.push_back(q[3]);
    }
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(V), verts.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(V), nullptr);
    glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog, "uResolution"), w, h);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verts.size()));
    glUseProgram(0);
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
}

void draw(float fps, float frameTimeMs, float zoom, float offX, float offY, int w, int h) {
    char hud1[256], hud2[256];
    std::snprintf(hud1, sizeof(hud1), "FPS: %.1f | Zoom: %.2f | Offset: (%.3f, %.3f)", fps, zoom, offX, offY);
    std::snprintf(hud2, sizeof(hud2), "Frame Time: %.2f ms", frameTimeMs);
    drawText(hud1, 10.0f, 20.0f, (float)w, (float)h);
    drawText(hud2, 10.0f, 50.0f, (float)w, (float)h);
}

void cleanup() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(prog);
    glDisable(GL_BLEND);
}

} // namespace Hud
