// MAUS:
// Datei: src/warzenschwein_overlay.cpp
// 🐭 Maus-Kommentar: Vollständig gekapselt wie HeatmapOverlay – keine GL-State-Leaks.
// 🦦 Otter: Orphaning + SubData, deterministischer Draw, ASCII-Logs nur bei Bedarf. (Bezug zu Otter)
// 🐑 Schneefuchs: Pixel-Space im CPU-Pfad + px->NDC im VS; getrennte X/Y-Skalierung fixiert Aspect Ratio. (Bezug zu Schneefuchs)

#pragma warning(push)
#pragma warning(disable: 4100)

#include "pch.hpp"
#include "warzenschwein_overlay.hpp"
#include "warzenschwein_fontdata.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "luchs_log_host.hpp"
#include <vector>
#include <string>
#include <algorithm>

namespace WarzenschweinOverlay {

constexpr int glyphW = 8, glyphH = 12; // Bitmap-Font: 8x12 Pixel pro Glyph

// GL objects & CPU buffers
static GLuint vao = 0, vbo = 0, shader = 0;
static std::vector<float> vertices;   // interleaved: x,y,r,g,b  (glyphs)
static std::vector<float> background; // interleaved: x,y,r,g,b  (panel)
static std::string currentText;
static bool visible = true;

// cached uniform locations
static GLint uViewportPxLoc = -1;
static GLint uHudScaleLoc   = -1;

// ----------------------------------------------------------------------------
// Shader: Wir liefern Positionen in PIXELKOORDINATEN (Origin: top-left).
// Der Vertex-Shader rechnet erst am Ende korrekt nach NDC um (getrennt X/Y).
// Dadurch gibt es kein „Gestaucht“-Problem mehr.
// ----------------------------------------------------------------------------
static const char* vsSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPosPx;   // pixel space, origin top-left
layout(location = 1) in vec3 aColor;
uniform vec2 uViewportPx;              // (width, height) in pixels
uniform vec2 uHudScale;                // optional fine-tune scale per axis (default 1,1)
out vec3 vColor;

vec2 toNdc(vec2 px, vec2 vp) {
    float x = (px.x / vp.x) * 2.0 - 1.0;
    float y = 1.0 - (px.y / vp.y) * 2.0; // invert Y for top-left origin
    return vec2(x, y);
}

void main() {
    vec2 scaled = aPosPx * uHudScale;
    vec2 ndc    = toNdc(scaled, uViewportPx);
    gl_Position = vec4(ndc, 0.0, 1.0);
    vColor      = aColor;
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

// ---- Shader Utils -----------------------------------------------------------
static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    if (!s) {
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[WS-SHADER] glCreateShader failed (type=%u)", (unsigned)type);
        return 0;
    }
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[2048] = {0};
        glGetShaderInfoLog(s, (GLsizei)sizeof(buf), nullptr, buf);
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[WS-SHADER] Compilation failed (%s): %s",
                           type == GL_VERTEX_SHADER ? "Vertex" : "Fragment", buf);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint link(GLuint vs, GLuint fs) {
    if (vs == 0 || fs == 0) return 0;
    GLuint prog = glCreateProgram();
    if (!prog) {
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[WS-SHADER] glCreateProgram failed");
        return 0;
    }
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint linked = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char buf[2048] = {0};
        glGetProgramInfoLog(prog, (GLsizei)sizeof(buf), nullptr, buf);
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[WS-SHADER] Link failed: %s", buf);
        glDeleteProgram(prog);
        prog = 0;
    }
    return prog;
}

// ---- Init -------------------------------------------------------------------
static void initGL() {
    if (vao != 0) return;

    GLuint vs = compile(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsSrc);
    shader = link(vs, fs);
    if (shader == 0) {
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[WS] initGL failed (shader=0) - overlay disabled");
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

    // cache uniform locations
    uViewportPxLoc = glGetUniformLocation(shader, "uViewportPx");
    uHudScaleLoc   = glGetUniformLocation(shader, "uHudScale");

    if constexpr (Settings::debugLogging)
        LUCHS_LOG_HOST("[WS] initGL ok: VAO=%u VBO=%u Program=%u uViewportPxLoc=%d uHudScaleLoc=%d",
                       vao, vbo, shader, (int)uViewportPxLoc, (int)uHudScaleLoc);
}

// ---- Background Quad (pixel space) -----------------------------------------
static void buildBackground(float x0, float y0, float x1, float y1) {
    const float bg[3] = {0.10f, 0.10f, 0.10f};
    const float quad[6][5] = {
        {x0, y0, bg[0], bg[1], bg[2]},
        {x1, y0, bg[0], bg[1], bg[2]},
        {x1, y1, bg[0], bg[1], bg[2]},
        {x0, y0, bg[0], bg[1], bg[2]},
        {x1, y1, bg[0], bg[1], bg[2]},
        {x0, y1, bg[0], bg[1], bg[2]},
    };
    background.insert(background.end(), &quad[0][0], &quad[0][0] + 6 * 5);
}

// Otter: Symmetric padding; Schneefuchs: Glyph-Advance stabil, px-sicher.
// WICHTIG: Wir generieren ALLE Koordinaten in PIXELN (origin top-left).
void generateOverlayQuads(const std::string& text,
                          std::vector<float>& vertexOut,
                          std::vector<float>& backgroundOut,
                          const RendererState& ctx)
{
    (void)ctx; // nur für mögliche künftige Layout-Regeln; aktuell nicht benötigt
    vertexOut.clear();
    backgroundOut.clear();

    // Skalierung der HUD-Glyphen in Pixeln (Otter: Settings steuert Größe)
    const float scalePx = std::max(1.0f, Settings::hudPixelSize); // Schneefuchs: min 1.0

    // Anker oben-links in PIXELN (Außenabstand zum Fensterrand)
    const float marginX = 16.0f;
    const float marginY = 16.0f;
    const float x0 = marginX;
    const float y0 = marginY;

    // Zeilen splitten
    std::vector<std::string> lines;
    {
        std::string cur;
        cur.reserve(64);
        for (char c : text) {
            if (c == '\n') { lines.push_back(cur); cur.clear(); }
            else           { cur += c; }
        }
        if (!cur.empty()) lines.push_back(cur);
    }

    // Content-Box in PIXELN
    std::size_t maxW = 0;
    for (const auto& l : lines) maxW = std::max(maxW, l.size());
    const float advX   = (glyphW + 1) * scalePx; // 1px tracking
    const float advY   = (glyphH + 2) * scalePx; // 2px line gap
    const float boxW   = static_cast<float>(maxW) * advX;
    const float boxH   = static_cast<float>(lines.size()) * advY;

    // Symmetrisches Innenpadding (Pixel)
    const float padL = 12.0f, padR = 12.0f;
    const float padT = 8.0f,  padB = 8.0f;

    // Hintergrund-Panel (pixel space, y nach unten)
    const float bgX0 = x0 - padL;
    const float bgY0 = y0 - padT;
    const float bgX1 = x0 + boxW + padR;
    const float bgY1 = y0 + boxH + padB;
    buildBackground(bgX0, bgY0, bgX1, bgY1);

    // Glyphen zeichnen
    const float r = 1.0f, g = 0.8f, b = 0.3f;

    for (std::size_t row = 0; row < lines.size(); ++row) {
        const std::string& line = lines[row];
        const float yBase = y0 + row * advY; // top of line in pixel space

        for (std::size_t col = 0; col < line.size(); ++col) {
            const auto& glyph = WarzenschweinFont::get(line[col]);
            const float xBase = x0 + col * advX;

            for (int gy = 0; gy < glyphH; ++gy) {
                const uint8_t bits = glyph[gy];
                for (int gx = 0; gx < glyphW; ++gx) {
                    if ((bits >> (7 - gx)) & 1) {
                        const float x = xBase + gx * scalePx;
                        const float y = yBase + gy * scalePx;
                        const float quad[6][5] = {
                            {x,               y,               r, g, b},
                            {x + scalePx,     y,               r, g, b},
                            {x + scalePx,     y + scalePx,     r, g, b},
                            {x,               y,               r, g, b},
                            {x + scalePx,     y + scalePx,     r, g, b},
                            {x,               y + scalePx,     r, g, b},
                        };
                        vertexOut.insert(vertexOut.end(), &quad[0][0], &quad[0][0] + 6 * 5);
                    }
                }
            }
        }
    }
}

// ---- Draw -------------------------------------------------------------------
void drawOverlay(RendererState& ctx) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[WS] drawOverlay: visible=%d empty=%d viewport=%dx%d",
                       visible ? 1 : 0, currentText.empty() ? 1 : 0,
                       (int)ctx.width, (int)ctx.height);
    }
    if (!Settings::warzenschweinOverlayEnabled || !visible || currentText.empty())
        return;

    initGL();
    if (shader == 0) return;

    generateOverlayQuads(currentText, vertices, background, ctx);

    // GL-State sichern
    GLint  prevVAO = 0, prevArray = 0, prevProg = 0;
    GLboolean wasDepth = GL_FALSE, wasBlend = GL_FALSE;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArray);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProg);
    glGetBooleanv(GL_DEPTH_TEST, &wasDepth);
    glGetBooleanv(GL_BLEND, &wasBlend);

    // HUD-State setzen
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(shader);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Set uniforms (viewport + optional per-axis scale = 1,1)
    if (uViewportPxLoc >= 0) glUniform2f(uViewportPxLoc, (float)ctx.width, (float)ctx.height);
    if (uHudScaleLoc   >= 0) glUniform2f(uHudScaleLoc,   1.0f, 1.0f);

    // Background upload (Orphaning + SubData)
    if (!background.empty()) {
        const GLsizeiptr bytes = (GLsizeiptr)(background.size() * sizeof(float));
        glBufferData(GL_ARRAY_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, background.data());
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(background.size() / 5));
    }

    // Glyph upload (Orphaning + SubData)
    if (!vertices.empty()) {
        const GLsizeiptr bytes = (GLsizeiptr)(vertices.size() * sizeof(float));
        glBufferData(GL_ARRAY_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, vertices.data());
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 5));
    }

    // GL-State restaurieren
    if (!wasBlend) glDisable(GL_BLEND);
    if (wasDepth)  glEnable(GL_DEPTH_TEST);
    glBindBuffer(GL_ARRAY_BUFFER, prevArray);
    glBindVertexArray((GLuint)prevVAO);
    glUseProgram((GLuint)prevProg);

    if constexpr (Settings::debugLogging) {
        const GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            LUCHS_LOG_HOST("[WS-GL] glGetError=0x%04X", err);
        }
    }
}

// ---- API --------------------------------------------------------------------
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
    uViewportPxLoc = -1;
    uHudScaleLoc   = -1;
}

} // namespace WarzenschweinOverlay

#pragma warning(pop)
