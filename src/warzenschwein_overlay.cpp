// Datei: src/warzenschwein_overlay.cpp
// 🐭 Maus-Kommentar: Vollständig gekapselt wie HeatmapOverlay - Textzustand lokal. drawOverlay prüft `visible` und `currentText`, Shader/VAO/VBO intern verwaltet. Kein Zugriff auf ctx außer Zoom. Otter: saubere Trennung, Schneefuchs: klares Datenmodell.

#include "warzenschwein_overlay.hpp"
#include "warzenschwein_fontdata.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "luchs_log_host.hpp"
#include <GL/glew.h>
#include <vector>
#include <string>

namespace WarzenschweinOverlay {

constexpr int glyphW = 8, glyphH = 12;

static GLuint vao = 0, vbo = 0, shader = 0;
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
        LUCHS_LOG_HOST("[ShaderError] Warzenschwein %s: %s",
                       type == GL_VERTEX_SHADER ? "Vertex" : "Fragment", buf);
        glDeleteShader(s);
        return 0;
    }

    return s;
}

static void initGL() {
    if (vao != 0) return;

    GLuint vs = compile(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsSrc);

    if (!vs || !fs) {
        LUCHS_LOG_HOST("[FATAL] WarzenschweinOverlay: Shader-Kompilierung fehlgeschlagen");
        return;
    }

    shader = glCreateProgram();
    glAttachShader(shader, vs);
    glAttachShader(shader, fs);
    glLinkProgram(shader);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint linked = GL_FALSE;
    glGetProgramiv(shader, GL_LINK_STATUS, &linked);
    if (!linked) {
        char buf[512];
        glGetProgramInfoLog(shader, 512, nullptr, buf);
        LUCHS_LOG_HOST("[ShaderError] Warzenschwein Link: %s", buf);
        glDeleteProgram(shader);
        shader = 0;
        return;
    }

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

// Otter: symmetric padding for top/bottom; clean, deterministic math. (Bezug zu Otter)
// Schneefuchs: keep glyph advance stable; only shift start Y by deltaTop. (Bezug zu Schneefuchs)
void generateOverlayQuads(const std::string& text,
                          std::vector<float>& vertexOut,
                          std::vector<float>& backgroundOut,
                          const RendererState& ctx)
{
    vertexOut.clear();
    backgroundOut.clear();

    // --- Screen pixel to NDC scale ---
    const float px  = Settings::hudPixelSize;   // 1 HUD pixel in screen pixels
    const float pxX = 2.0f / ctx.width;
    const float pxY = 2.0f / ctx.height;

    // --- Anchor (outer top-left of content area, before padding) ---
    const float x0 = -1.0f + 16.0f * pxX;
    const float y0 =  1.0f - 16.0f * pxY;

    // --- Split text into lines ---
    std::vector<std::string> lines;
    {
        std::string cur;
        for (char c : text) {
            if (c == '\n') { lines.push_back(cur); cur.clear(); }
            else           { cur += c; }
        }
        if (!cur.empty()) lines.push_back(cur);
    }

    // --- Compute content box (in HUD pixels) ---
    std::size_t maxW = 0;
    for (const auto& l : lines) maxW = std::max(maxW, l.size());

    // Letter/line spacing are baked like before: +1 horiz, +2 vert; +2 overall border fudge.
    const float boxW = (maxW * (glyphW + 1) + 2) * px;
    const float boxH = (static_cast<float>(lines.size()) * (glyphH + 2) + 2) * px;

    // --- NEW: symmetric inner padding (top == bottom), left/right as before ---
    // We liked the *bottom* margin visually; mirror it to the top.
    // Choose 4 px HUD-padding for top/bottom, 1 px left/right (keeps previous look).
    const float padL = 1.0f * px;
    const float padR = 1.0f * px;
    const float padT = 4.0f * px;
    const float padB = 4.0f * px;

    // Background quad with symmetric padding
    const float bgX0 = x0 - padL;
    const float bgY0 = y0 + padT;
    const float bgX1 = x0 + boxW + padR;
    const float bgY1 = y0 - boxH - padB;
    buildBackground(bgX0, bgY0, bgX1, bgY1);

    // Shift first text baseline down by the *additional* top padding delta.
    // Previously top padding was 1*px; now it is padT. Keep advance/grid identical.
    const float deltaTop = (padT - 1.0f * px);

    // --- Draw glyph quads (unchanged advance, only baseline shift) ---
    const float r = 1.0f, g = 0.8f, b = 0.3f;

    for (std::size_t row = 0; row < lines.size(); ++row) {
        const std::string& line = lines[row];
        const float yBase = (y0 - deltaTop) - static_cast<float>(row) * (glyphH + 2) * px;

        for (std::size_t col = 0; col < line.size(); ++col) {
            const auto& glyph = WarzenschweinFont::get(line[col]);
            const float xBase = x0 + static_cast<float>(col) * (glyphW + 1) * px;

            for (int gy = 0; gy < glyphH; ++gy) {
                const uint8_t bits = glyph[gy];
                for (int gx = 0; gx < glyphW; ++gx) {
                    if ((bits >> (7 - gx)) & 1) {
                        const float x = xBase + gx * px;
                        const float y = yBase - gy * px;
                        const float quad[6][5] = {
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
        LUCHS_LOG_HOST("[WZ] drawOverlay called, visible=%d, currentText.empty=%d",
                       visible ? 1 : 0,
                       currentText.empty() ? 1 : 0);
    }

    if (!Settings::warzenschweinOverlayEnabled || !visible || currentText.empty())
        return;

    initGL();
    if (!shader) return;

    generateOverlayQuads(currentText, vertices, background, ctx);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(shader);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER, background.size() * sizeof(float), background.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(background.size() / 5));

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 5));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    if (Settings::debugLogging) {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            LUCHS_LOG_HOST("[WS-GL] glGetError = 0x%04X", err);
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
