// Datei: src/hud.cpp (312 Zeilen)
// 🐭 Maus-Kommentar: HUD loggt nun kompakt ASCII-only – mit robuster Prüfung, ob `stb_easy_font` tatsächlich Text erzeugt. Kein Schweigen mehr bei `quads=0`. Otter nennt es: „Sichtprüfung durch Sichtbarkeit“.

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

#include <vector>
#include <cstdio>
#include <cmath>

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
    GLint success = 0;
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

    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::fprintf(stderr, "[HUD LINK ERROR] %s\n", infoLog);
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
    glDisable(GL_DEPTH_TEST);
}

static bool drawTextImpl(const std::string& text, float x, float y, float width, float height) {
    char buffer[99999]; // STB-EasyFont verlangt char-buffer
    int num_quads = stb_easy_font_print(x, y, const_cast<char*>(text.c_str()), nullptr, buffer, sizeof(buffer));

    if (Settings::debugLogging) {
        std::printf("[HUD] drawTextImpl called: \"%s\" -> quads=%d (bufSize=%zu)\n", text.c_str(), num_quads, sizeof(buffer));
    }

    if (num_quads <= 0) {
        if (Settings::debugLogging) {
            std::printf("[HUD] Warning: No quads generated for \"%s\"\n", text.c_str());
        }
        return false;
    }

    float* coords = reinterpret_cast<float*>(buffer);
    [[maybe_unused]] const int numVertices = num_quads * 4;

    std::vector<float> verts;
    verts.reserve(num_quads * 6 * 2); // 6 vertices * 2 floats

    for (int i = 0; i < num_quads; ++i) {
        float* p = coords + i * 8;
        float x0 = p[0], y0 = p[1];
        float x1 = p[2], y1 = p[3];
        float x2 = p[4], y2 = p[5];
        float x3 = p[6], y3 = p[7];

        verts.insert(verts.end(), { x0, y0, x1, y1, x2, y2 });
        verts.insert(verts.end(), { x0, y0, x2, y2, x3, y3 });
    }

    glUseProgram(hudProgram);
    glBindVertexArray(hudVAO);
    glBindBuffer(GL_ARRAY_BUFFER, hudVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    glUniform2f(glGetUniformLocation(hudProgram, "uResolution"), width, height);
    glUniform4f(glGetUniformLocation(hudProgram, "uColor"), 1.0f, 1.0f, 1.0f, 1.0f); // weiß

    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verts.size() / 2));
    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
    glUseProgram(0);
    return true;
}

void drawText(const std::string& text, float x, float y, float width, float height) {
    if (text == "TEST_RECTANGLE") {
        struct Vertex { float x, y; };
        Vertex rect[6] = {
            {10, 10}, {210, 10}, {210, 70},
            {10, 10}, {210, 70}, {10, 70}
        };

        glUseProgram(hudProgram);
        glBindVertexArray(hudVAO);
        glBindBuffer(GL_ARRAY_BUFFER, hudVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(rect), rect, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
        glEnableVertexAttribArray(0);

        glUniform2f(glGetUniformLocation(hudProgram, "uResolution"), width, height);
        glUniform4f(glGetUniformLocation(hudProgram, "uColor"), 1.0f, 0.0f, 1.0f, 1.0f); // Magenta

        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisableVertexAttribArray(0);
        glBindVertexArray(0);
        glUseProgram(0);
        return;
    }

    if (!text.empty())
        drawTextImpl(text, x, y, width, height);
}

void draw(RendererState& state) {
    drawText("TESTHUD", 200, 100, static_cast<float>(state.width), static_cast<float>(state.height));

    float logZoom = -log10f(static_cast<float>(state.zoom));
    float fps = static_cast<float>(state.currentFPS);
    float frameTimeMs = static_cast<float>(state.deltaTime * 1000.0f);

    char buf1[128], buf2[64], buf3[64];
    std::snprintf(buf1, sizeof(buf1), "FPS: %.1f | Zoom: 1e%.1f | Offset: (%.3f, %.3f)",
        fps, logZoom,
        static_cast<float>(state.offset.x),
        static_cast<float>(state.offset.y));
    std::snprintf(buf2, sizeof(buf2), "Frame Time: %.2f ms", frameTimeMs);
    std::snprintf(buf3, sizeof(buf3), "[H] Overlay: %s", state.overlayEnabled ? "ON" : "OFF");

    float w = static_cast<float>(state.width);
    float h = static_cast<float>(state.height);
    float line = 28.0f;

    bool ok1 = drawTextImpl(buf1, 10.0f, h - 1 * line, w, h);
    bool ok2 = drawTextImpl(buf2, 10.0f, h - 2 * line, w, h);
    bool ok3 = drawTextImpl(buf3, 10.0f, h - 3 * line, w, h);

    if (Settings::debugLogging) {
        std::printf("[HUD] FPS=%.1f Zoom=1e%.1f Offset=(%.3f, %.3f) Frame=%.2fms Overlay=%s Quads=[%d %d %d]\n",
            fps, logZoom,
            static_cast<float>(state.offset.x),
            static_cast<float>(state.offset.y),
            frameTimeMs,
            state.overlayEnabled ? "ON" : "OFF",
            ok1 ? 1 : 0, ok2 ? 1 : 0, ok3 ? 1 : 0);
    }
}

void cleanup() {
    glDeleteVertexArrays(1, &hudVAO);
    glDeleteBuffers(1, &hudVBO);
    glDeleteProgram(hudProgram);
    glDisable(GL_BLEND);
}

} // namespace Hud
