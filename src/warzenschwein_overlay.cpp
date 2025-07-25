// 🐭 Maus-Kommentar: pixelSize jetzt dynamisch über ctx.zoom wie bei Heatmap. Einheitlich, stabil, keine Frustquelle mehr. Schneefuchs: „Zoom-Faktor muss rein.“

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
    ctx.warzenschweinOverlayEnabled = !ctx.warzenschweinOverlayEnabled;
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
    if (!ctx.warzenschweinOverlayEnabled) return;
    if (currentLines.empty()) return;

    const float pixelSize = 0.0025f / static_cast<float>(ctx.zoom); // dynamisch

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

    // Hintergrundbox
    size_t maxWidth = 0;
    for (const auto& line : currentLines)
        if (line.length() > maxWidth) maxWidth = line.length();

    float boxW = (maxWidth * (glyphWidth + 1) + 2) * pixelSize;
    float boxH = (currentLines.size() * (glyphHeight + 2) + 2) * pixelSize;

    float x0 = startX - pixelSize;
    float y0 = startY + pixelSize;
    float x1 = startX + boxW;
    float y1 = startY - boxH;

    float bg[6][5] = {
        {x0, y0, 0.2f, 0.2f, 0.2f},
        {x1, y0, 0.2f, 0.2f, 0.2f},
        {x1, y1, 0.2f, 0.2f, 0.2f},
        {x0, y0, 0.2f, 0.2f, 0.2f},
        {x1, y1, 0.2f, 0.2f, 0.2f},
        {x0, y1, 0.2f, 0.2f, 0.2f},
    };
    for (auto& v : bg)
        vertices.insert(vertices.end(), v, v + 5);

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
