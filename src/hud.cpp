// Datei: src/hud.cpp
// Zeilen: 230
// 🐭 Maus-Kommentar: HUD-Overlay mit Textanzeige via STB-Easy-Font und GLSL-Shadern. Zeigt FPS, Offset, Zoom-Faktor als wissenschaftliche Potenz sowie Overlay-Status. Schneefuchs: „Alles sichtbar, nichts verborgen.“
// Otter-Fix: Shader-Linking wird nun geprüft. Float-Casts überall für MSVC /W4 /WX. Shader-Fehlerausgabe erweitert.

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

#include <cmath>

namespace Hud {

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
    FragColor = vec4(1.0, 1.0, 1.0, 1.0); // Weißes HUD mit voller Deckkraft
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
        fprintf(stderr, "[HUD SHADER ERROR]\n%s\n", infoLog);
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
        fprintf(stderr, "[HUD LINK ERROR]\n%s\n", infoLog);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

void init() {
    glGenVertexArrays(1, &hudVAO);
    glGenBuffers(1, &hudVBO);
    hudProgram = createHUDProgram();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void drawText(const std::string& text, float x, float y, float width, float height) {
    if (text.empty()) return;

    char buffer[99999];
    int num_quads = stb_easy_font_print(x, y, const_cast<char*>(text.c_str()), nullptr, buffer, sizeof(buffer));

    struct Vertex { float x, y; };
    std::vector<Vertex> vertices;
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

    glBindVertexArray(hudVAO);
    glBindBuffer(GL_ARRAY_BUFFER, hudVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);

    glUseProgram(hudProgram);
    glUniform2f(glGetUniformLocation(hudProgram, "uResolution"), width, height);

    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));

    glUseProgram(0);
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
}

void draw(RendererState& state) {
    char hudText1[256];
    char hudText2[256];
    char hudText3[64];

    float logZoom = -log10f(static_cast<float>(state.zoom));
    float fps = static_cast<float>(state.currentFPS);
    float frameTimeMs = static_cast<float>(state.deltaTime * 1000.0);

    std::snprintf(hudText1, sizeof(hudText1),
                  "FPS: %.1f | Zoom: 1e%.1f | Offset: (%.3f, %.3f)",
                  fps, logZoom,
                  static_cast<float>(state.offset.x),
                  static_cast<float>(state.offset.y));
    std::snprintf(hudText2, sizeof(hudText2),
                  "Frame Time: %.2f ms", frameTimeMs);
    std::snprintf(hudText3, sizeof(hudText3),
                  "[H] Overlay: %s", state.overlayEnabled ? "ON" : "OFF");

    drawText(hudText1, 10.0f, 20.0f, static_cast<float>(state.width), static_cast<float>(state.height));
    drawText(hudText2, 10.0f, 50.0f, static_cast<float>(state.width), static_cast<float>(state.height));
    drawText(hudText3, 10.0f, 80.0f, static_cast<float>(state.width), static_cast<float>(state.height));
}

void cleanup() {
    glDeleteVertexArrays(1, &hudVAO);
    glDeleteBuffers(1, &hudVBO);
    glDeleteProgram(hudProgram);
    glDisable(GL_BLEND);
}

} // namespace Hud
