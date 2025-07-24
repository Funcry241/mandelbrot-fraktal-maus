// Datei: src/warzenschwein_overlay.cpp
// 🐭 Maus-Kommentar: Warzenschwein nutzt jetzt direkt die kompakte Font-Map aus `warzenschwein_font.hpp`. Kein eigener Glyph-Cache mehr, volle Wiederverwendbarkeit. Schneefuchs: „Ein Overlay, ein Font – Ende der Vervielfachung!“

#include "pch.hpp"
#include "warzenschwein_overlay.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "warzenschwein_fontdata.hpp"
#include <string>
#include <vector>
#include <cstdio>

namespace WarzenschweinOverlay {

static GLuint vao = 0;
static GLuint vbo = 0;
static GLuint shader = 0;
static std::vector<std::string> currentLines;

constexpr int glyphWidth = 8;
constexpr int glyphHeight = 12;
constexpr float pixelSize = 0.005f; // World-Space-Größe pro Pixel

static const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;
out vec3 vColor;
uniform vec2 uOffset;
uniform vec2 uScale;
void main() {
    vec2 pos = aPos * uScale + uOffset;
    gl_Position = vec4(pos, 0.0, 1.0);
    vColor = aColor;
}
)GLSL";

static const char* fragmentShaderSrc = R"GLSL(
#version 430 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
)GLSL";

static GLuint compile(GLenum type, const char* src) {
    GLuint localShader = glCreateShader(type);
    glShaderSource(localShader, 1, &src, nullptr);
    glCompileShader(localShader);
    GLint success = 0;
    glGetShaderiv(localShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar log[1024];
        glGetShaderInfoLog(localShader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[SHADER ERROR] Compilation failed: %s\n", log);
    }
    return localShader;
}

static GLuint createShaderProgram() {
    GLuint vs = compile(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint success = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::fprintf(stderr, "[SHADER ERROR] Linking failed: %s\n", log);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

void toggle(RendererState& ctx) {
    ctx.overlayEnabled = !ctx.overlayEnabled;
}

void cleanup() {
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (shader) glDeleteProgram(shader);
    vao = vbo = shader = 0;
    currentLines.clear();
}

void setText(const std::string& text, int /*x*/, int /*y*/) {
    currentLines.clear();
    std::string line;
    for (char c : text) {
        if (c == '\n') {
            currentLines.push_back(line);
            line.clear();
        } else {
            line += c;
        }
    }
    if (!line.empty()) {
        currentLines.push_back(line);
    }
}

void drawOverlay(RendererState& ctx) {
    if (!ctx.overlayEnabled || currentLines.empty()) return;

    if (vao == 0) {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        shader = createShaderProgram();
    }

    glUseProgram(shader);

    std::vector<float> vertices;
    float startX = -1.0f + 0.02f;
    float startY =  1.0f - 0.02f;
    float r = 1.0f, g = 0.8f, b = 0.3f;

    for (size_t lineIdx = 0; lineIdx < currentLines.size(); ++lineIdx) {
        float yBase = startY - lineIdx * (glyphHeight + 2) * pixelSize;
        const std::string& line = currentLines[lineIdx];
        for (size_t col = 0; col < line.size(); ++col) {
            char c = line[col];
            const auto& bitmap = WarzenschweinFont::get(c);
            if (bitmap == WarzenschweinFont::Glyph{}) continue;

            float xBase = startX + col * (glyphWidth + 1) * pixelSize;
            for (int row = 0; row < glyphHeight; ++row) {
                uint8_t bits = bitmap[row];
                for (int bit = 0; bit < glyphWidth; ++bit) {
                    if ((bits >> (7 - bit)) & 1) {
                        float x = xBase + bit * pixelSize;
                        float y = yBase - row * pixelSize;
                        float quad[6][5] = {
                            {x,             y,             r, g, b},
                            {x + pixelSize, y,             r, g, b},
                            {x + pixelSize, y - pixelSize, r, g, b},
                            {x,             y,             r, g, b},
                            {x + pixelSize, y - pixelSize, r, g, b},
                            {x,             y - pixelSize, r, g, b},
                        };
                        for (auto& v : quad)
                            vertices.insert(vertices.end(), v, v + 5);
                    }
                }
            }
        }
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));

    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 5));

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
    glUseProgram(0);
}

} // namespace WarzenschweinOverlay
