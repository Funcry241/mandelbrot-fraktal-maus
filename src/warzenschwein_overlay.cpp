// Datei: src/warzenschwein_overlay.cpp
// 🐭 Maus-Kommentar: Vollständig gekapselt wie HeatmapOverlay – Textzustand lokal. drawOverlay prüft `visible` und `currentText`, Shader/VAO/VBO intern verwaltet. Kein Zugriff auf ctx außer Zoom. Otter: saubere Trennung, Schneefuchs: klares Datenmodell.

#include "warzenschwein_overlay.hpp"
#include "warzenschwein_fontdata.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include <GL/glew.h>
#include <vector>
#include <string>

namespace WarzenschweinOverlay {

constexpr int glyphW = 8, glyphH = 12;

// 🔒 Interner Zustand
static GLuint vao = 0;
static GLuint vbo = 0;
static GLuint shader = 0;
static std::vector<float> vertices;
static std::vector<float> background;
static std::string currentText;
static bool visible = true;

static const char* vsSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;
out vec3 vColor;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
)GLSL";

static const char* fsSrc = R"GLSL(
#version 430 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
)GLSL";

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512];
        glGetShaderInfoLog(s, 512, nullptr, buf);
        std::fprintf(stderr, "[ShaderError] %s: %s\n", (type == GL_VERTEX_SHADER ? "Vertex" : "Fragment"), buf);
    }
    return s;
}

static void initGL() {
    if (vao != 0) return;
    shader = glCreateProgram();
    GLuint vs = compile(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsSrc);
    glAttachShader(shader, vs);
    glAttachShader(shader, fs);
    glLinkProgram(shader);
    glDeleteShader(vs);
    glDeleteShader(fs);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

static void buildBackground(float x0, float y0, float x1, float y1) {
    float bgColor[3] = {0.10f, 0.10f, 0.10f};
    float quad[6][5] = {
        {x0, y0, bgColor[0], bgColor[1], bgColor[2]},
        {x1, y0, bgColor[0], bgColor[1], bgColor[2]},
        {x1, y1, bgColor[0], bgColor[1], bgColor[2]},
        {x0, y0, bgColor[0], bgColor[1], bgColor[2]},
        {x1, y1, bgColor[0], bgColor[1], bgColor[2]},
        {x0, y1, bgColor[0], bgColor[1], bgColor[2]},
    };
    background.insert(background.end(), &quad[0][0], &quad[0][0] + 6 * 5);
}

void generateOverlayQuads(
    const std::string& text,
    std::vector<float>& vertexOut,
    std::vector<float>& backgroundOut,
    const RendererState& ctx
) {
    vertexOut.clear();
    backgroundOut.clear();
    const float px = Settings::hudPixelSize;
    const float pxX = 2.0f / ctx.width;
    const float pxY = 2.0f / ctx.height;
    const float x0 = -1.0f + 16.0f * pxX;
    const float y0 =  1.0f - 16.0f * pxY;

    std::vector<std::string> lines;
    std::string cur;
    for (char c : text) {
        if (c == '\n') { lines.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) lines.push_back(cur);

    size_t maxW = 0;
    for (const auto& l : lines) maxW = std::max(maxW, l.size());
    float boxW = (maxW * (glyphW + 1) + 2) * px;
    float boxH = (lines.size() * (glyphH + 2) + 2) * px;
    buildBackground(x0 - px, y0 + px, x0 + boxW, y0 - boxH);

    float r = 1.0f, g = 0.8f, b = 0.3f;

    for (size_t row = 0; row < lines.size(); ++row) {
        const std::string& line = lines[row];
        float yBase = y0 - row * (glyphH + 2) * px;
        for (size_t col = 0; col < line.size(); ++col) {
            const auto& glyph = WarzenschweinFont::get(line[col]);
            float xBase = x0 + col * (glyphW + 1) * px;
            for (int gy = 0; gy < glyphH; ++gy) {
                uint8_t bits = glyph[gy];
                for (int gx = 0; gx < glyphW; ++gx) {
                    if ((bits >> (7 - gx)) & 1) {
                        float x = xBase + gx * px;
                        float y = yBase - gy * px;
                        float quad[6][5] = {
                            {x,       y,       r, g, b},
                            {x + px,  y,       r, g, b},
                            {x + px,  y - px,  r, g, b},
                            {x,       y,       r, g, b},
                            {x + px,  y - px,  r, g, b},
                            {x,       y - px,  r, g, b},
                        };
                        vertexOut.insert(vertexOut.end(), &quad[0][0], &quad[0][0] + 6 * 5);
                    }
                }
            }
        }
    }
}

void drawOverlay(RendererState& ctx) {
     if (Settings::debugLogging) {
        printf("[WS-Precheck] visible=%d | empty=%d\n",
            static_cast<int>(visible),
            static_cast<int>(currentText.empty())
        );
    }

    if (!Settings::warzenschweinOverlayEnabled) return;
    if (!visible || currentText.empty()) return;

    initGL();
    generateOverlayQuads(currentText, vertices, background, ctx);

    if (Settings::debugLogging) {
        printf("[WS-Overlay] Visible=%d | TextLen=%zu | Verts=%zu | BG=%zu | Zoom=%.3f\n",
            visible,
            currentText.length(),
            vertices.size(),
            background.size(),
            ctx.zoom
        );
    }

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(shader);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER, background.size() * sizeof(float), background.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(background.size() / 5));

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(vertices.size() / 5));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    if (Settings::debugLogging) {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            printf("[WS-GL] glGetError = 0x%04X\n", err);
        }
    }
}

void toggle(RendererState&) {
    visible = !visible;
}

void setText(const std::string& text) {
    currentText = text;
}

void cleanup() {
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (shader) glDeleteProgram(shader);
    vao = vbo = shader = 0;
    vertices.clear();
    background.clear();
    currentText.clear();
    visible = false;
}

} // namespace WarzenschweinOverlay
