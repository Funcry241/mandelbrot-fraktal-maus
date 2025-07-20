// Datei: src/hud.cpp
// 🐭 Maus-Kommentar: strncpy_s eingebaut, keine Warnung mehr. Alles ASCII. Alles sichtbar. Otter: „Sauber.“

#include "pch.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4505)
#endif
#define STB_EASY_FONT_IMPLEMENTATION
#include "stb_easy_font.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "hud.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace Hud {

static const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
uniform vec2 uResolution;
void main() {
    vec2 pos = aPos / uResolution * 2.0 - 1.0;
    gl_Position = vec4(pos.x, -pos.y, 0.0, 1.0);
}
)GLSL";

static const char* fragmentShaderSrc = R"GLSL(
#version 430 core
out vec4 FragColor;
uniform vec4 uColor;
void main() {
    FragColor = uColor;
}
)GLSL";

static GLuint hudVAO = 0;
static GLuint hudVBO = 0;
static GLuint hudProgram = 0;

static GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::fprintf(stderr, "[HUD SHADER ERROR] %s\n", infoLog);
    }
    return shader;
}

static GLuint createHUDProgram() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint linkStatus = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (!linkStatus) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::fprintf(stderr, "[HUD LINK ERROR] %s\n", infoLog);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

static void sanitizeASCII(char* buffer) {
    for (size_t i = 0; buffer[i]; ++i) {
        unsigned char c = static_cast<unsigned char>(buffer[i]);
        if (c < 32 || c > 126) buffer[i] = '?';
    }
}

void init() {
    glGenVertexArrays(1, &hudVAO);
    glGenBuffers(1, &hudVBO);
    hudProgram = createHUDProgram();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
}

void drawText(const char* rawText, float x, float y, float width, float height, bool fallbackRect = false) {
    if (!rawText || !*rawText) return;

    char asciiText[1024];
    strncpy_s(asciiText, sizeof(asciiText), rawText, _TRUNCATE); // FIX!
    sanitizeASCII(asciiText);

    char buffer[65536];
    int num_quads = stb_easy_font_print(x, y, asciiText, nullptr, buffer, sizeof(buffer));
    if (num_quads <= 0 && !fallbackRect) return;

    struct Vertex { float x, y; };
    std::vector<Vertex> vertices;

    if (fallbackRect) {
        float w = 220.0f, h = 70.0f;
        Vertex rect[6] = {
            {x, y}, {x + w, y}, {x + w, y + h},
            {x, y}, {x + w, y + h}, {x, y + h}
        };
        vertices.insert(vertices.end(), std::begin(rect), std::end(rect));
    } else {
        vertices.reserve(num_quads * 6);
        for (int i = 0; i < num_quads; ++i) {
            unsigned char* quad = reinterpret_cast<unsigned char*>(buffer) + i * 64;
            Vertex v0 = *reinterpret_cast<Vertex*>(quad +  0);
            Vertex v1 = *reinterpret_cast<Vertex*>(quad + 16);
            Vertex v2 = *reinterpret_cast<Vertex*>(quad + 32);
            Vertex v3 = *reinterpret_cast<Vertex*>(quad + 48);
            vertices.push_back(v0); vertices.push_back(v1); vertices.push_back(v2);
            vertices.push_back(v0); vertices.push_back(v2); vertices.push_back(v3);
        }
    }

    glUseProgram(hudProgram);
    glBindVertexArray(hudVAO);
    glBindBuffer(GL_ARRAY_BUFFER, hudVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
    glEnableVertexAttribArray(0);

    glUniform2f(glGetUniformLocation(hudProgram, "uResolution"), width, height);
    glUniform4f(glGetUniformLocation(hudProgram, "uColor"),
        fallbackRect ? 1.0f : 1.0f,
        fallbackRect ? 0.0f : 1.0f,
        fallbackRect ? 1.0f : 1.0f,
        1.0f);

    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));

    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
    glUseProgram(0);

    if (Settings::debugLogging && !fallbackRect)
        std::printf("[HUD] \"%s\" Q=%d V=%zu\n", asciiText, num_quads, vertices.size());
}

void draw(RendererState& state) {
    const float left = 10.0f;
    const float lineHeight = 24.0f;
    const float baseY = static_cast<float>(state.height);

    char hudText1[256];
    char hudText2[256];
    char hudText3[64];

    float logZoom = -log10f(static_cast<float>(state.zoom));
    float fps = static_cast<float>(state.currentFPS);
    float frameTimeMs = static_cast<float>(state.deltaTime * 1000.0f);

    std::snprintf(hudText1, sizeof(hudText1),
        "FPS: %.1f | Zoom: 1e%.1f | Offset: (%.3f, %.3f)",
        fps, logZoom,
        static_cast<float>(state.offset.x),
        static_cast<float>(state.offset.y));
    std::snprintf(hudText2, sizeof(hudText2), "Frame Time: %.2f ms", frameTimeMs);
    std::snprintf(hudText3, sizeof(hudText3), "[H] Overlay: %s", state.overlayEnabled ? "ON" : "OFF");

    drawText(hudText1, left, baseY - 1 * lineHeight, static_cast<float>(state.width), static_cast<float>(state.height));
    drawText(hudText2, left, baseY - 2 * lineHeight, static_cast<float>(state.width), static_cast<float>(state.height));
    drawText(hudText3, left, baseY - 3 * lineHeight, static_cast<float>(state.width), static_cast<float>(state.height));

    if (Settings::debugLogging)
        drawText("DEBUG_RECT", 10.0f, 10.0f, static_cast<float>(state.width), static_cast<float>(state.height), true);
}

void cleanup() {
    glDeleteVertexArrays(1, &hudVAO);
    glDeleteBuffers(1, &hudVBO);
    glDeleteProgram(hudProgram);
    glDisable(GL_BLEND);
}

} // namespace Hud
