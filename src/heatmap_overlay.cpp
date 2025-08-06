// Datei: src/heatmap_overlay.cpp
// Zweck: Diagnose der Koordinatenumrechnung zwischen Heatmap-Tiles, Screen-Pixeln und Fraktal.
// Enthält: Mini-Overlay, Self-Check-Punkte, Max-Tile-Markierung im Hauptbild, Logging der Complex-Koordinate.

#pragma warning(push)
#pragma warning(disable: 4100)

#include "pch.hpp"
#include "heatmap_overlay.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_utils.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

// ============================================================================
// Kamera-/View-Adapter (RendererState-Felder projektabhängig, daher Makros)
// ----------------------------------------------------------------------------
// Wenn dein RendererState andere Namen hat, definiere beim Build z. B.:
//   /DRS_HAS_OFFSET /DRS_OFFSET_X_EXPR=ctx.centerX /DRS_OFFSET_Y_EXPR=ctx.centerY
//   /DRS_HAS_ZOOM   (dann wird RS_ZOOM(ctx) = ctx.zoom genutzt)
// Ohne diese Defines fallen wir auf 0/0/1 zurück (kompiliert immer, aber nur Diagnose).
// ============================================================================
#ifndef RS_OFFSET_X_EXPR
  #define RS_OFFSET_X_EXPR (0.0)
#endif
#ifndef RS_OFFSET_Y_EXPR
  #define RS_OFFSET_Y_EXPR (0.0)
#endif
#ifndef RS_ZOOM_EXPR
  #define RS_ZOOM_EXPR     (1.0)
#endif

#ifdef RS_HAS_OFFSET
  #define RS_OFFSET_X(ctx) (RS_OFFSET_X_EXPR)
  #define RS_OFFSET_Y(ctx) (RS_OFFSET_Y_EXPR)
#else
  #define RS_OFFSET_X(ctx) (0.0)
  #define RS_OFFSET_Y(ctx) (0.0)
#endif

#ifdef RS_HAS_ZOOM
  #define RS_ZOOM(ctx)     (RS_ZOOM_EXPR)
#else
  #define RS_ZOOM(ctx)     (1.0)
#endif

namespace HeatmapOverlay {

// ---------- GL State ----------
static GLuint overlayVAO = 0;
static GLuint overlayVBO = 0;
static GLuint overlayShader = 0;

static GLuint pointProg   = 0; // für Screen-/Overlay-Punkte
static GLuint pointVAO    = 0;
static GLuint pointVBO    = 0;

// ---------- Overlay Shader ----------
static const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;   // Kachelkoordinaten (x,y) in [0..tilesX],[0..tilesY]
layout(location = 1) in float aValue;
out float vValue;
uniform vec2 uOffset;  // NDC Offset des Mini-Overlays
uniform vec2 uScale;   // NDC Scale des Mini-Overlays
void main() {
    vec2 pos = aPos * uScale + uOffset;
    gl_Position = vec4(pos, 0.0, 1.0);
    vValue = aValue;
}
)GLSL";

static const char* fragmentShaderSrc = R"GLSL(
#version 430 core
in float vValue;
out vec4 FragColor;
vec3 colormap(float v) {
    float g = smoothstep(0.0, 1.0, v);
    return mix(vec3(0.08, 0.08, 0.10), vec3(1.0, 0.6, 0.2), g);
}
void main() {
    FragColor = vec4(colormap(clamp(vValue, 0.0, 1.0)), 0.85);
}
)GLSL";

// ---------- Minimaler Punkt-Shader (für Overlay- und Screen-Punkte) ----------
static const char* pointVS = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;   // generische Position (Einheit siehe uScale/uOffset)
layout(location=1) in vec3 aColor;
uniform vec2 uScale;               // wandelt aPos in NDC: pos_ndc = aPos*uScale + uOffset
uniform vec2 uOffset;
uniform float uPointSize;
out vec3 vColor;
void main(){
    vec2 pos = aPos * uScale + uOffset;
    gl_Position = vec4(pos, 0.0, 1.0);
    gl_PointSize = uPointSize;
    vColor = aColor;
}
)GLSL";

static const char* pointFS = R"GLSL(
#version 430 core
in vec3 vColor;
out vec4 FragColor;
void main(){ FragColor = vec4(vColor, 1.0); }
)GLSL";

// ---------- Hilfen ----------
static GLuint compile(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        LUCHS_LOG_HOST("[SHADER ERROR] Compilation failed: %s", log);
    }
    return shader;
}

static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint success = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        LUCHS_LOG_HOST("[SHADER ERROR] Linking failed: %s", log);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

static GLuint createOverlayProgram() {
    return linkProgram(compile(GL_VERTEX_SHADER,   vertexShaderSrc),
                       compile(GL_FRAGMENT_SHADER, fragmentShaderSrc));
}

static void ensurePointPipeline()
{
    if (pointProg) return;
    pointProg = linkProgram(compile(GL_VERTEX_SHADER, pointVS),
                            compile(GL_FRAGMENT_SHADER, pointFS));
    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);
}

// Zeichnet N Punkte mit aPos=(x,y) und aColor=(r,g,b) – Umrechnung nach NDC über uScale/uOffset.
// Erwartet: pts ist Array aus {x,y,r,g,b}, stride=5 floats.
static void drawPoints(const float* pts, int count, float scaleX, float scaleY, float offX, float offY, float pointSize)
{
    ensurePointPipeline();
    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*5*count, pts, GL_DYNAMIC_DRAW);

    glUseProgram(pointProg);
    glUniform2f(glGetUniformLocation(pointProg,"uScale"),  scaleX, scaleY);
    glUniform2f(glGetUniformLocation(pointProg,"uOffset"), offX, offY);
    glUniform1f(glGetUniformLocation(pointProg,"uPointSize"), pointSize);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(2*sizeof(float)));

    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, count);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
    glUseProgram(0);
}

// Self-Check: fünf eindeutige Punkte im Mini-Overlay – gleiche Transform wie Overlay.
// Jeder Punkt hat eine eigene Farbe und eine klare Position:
// - BL  = Bottom Left   (unten links)      -> Blau
// - BR  = Bottom Right  (unten rechts)     -> Grün
// - TR  = Top Right     (oben rechts)      -> Gelb
// - TL  = Top Left      (oben links)       -> Magenta
// - CTR = Center        (Bildmitte)        -> Rot
static void DrawHeatmapSelfCheck_OverlaySpace(int tilesX, int tilesY,
                                              float scaleX, float scaleY,
                                              float offsetX, float offsetY)
{
    const float pts[5][5] = {
        { 0.5f,              0.5f,               0.0f, 0.0f, 1.0f }, // BL  - unten links   - Blau
        { tilesX-0.5f,       0.5f,               0.0f, 1.0f, 0.0f }, // BR  - unten rechts  - Grün
        { tilesX-0.5f,       tilesY-0.5f,        1.0f, 1.0f, 0.0f }, // TR  - oben rechts   - Gelb
        { 0.5f,              tilesY-0.5f,        1.0f, 0.0f, 1.0f }, // TL  - oben links    - Magenta
        { tilesX*0.5f,       tilesY*0.5f,        1.0f, 0.0f, 0.0f }, // CTR - Bildmitte     - Rot
    };
    drawPoints(&pts[0][0], 5, scaleX, scaleY, offsetX, offsetY, 10.0f);
}

// Zeichnet einen Punkt im HAUPTBILD an Pixelzentrum (px,py).
static void DrawPoint_ScreenPixels(float px, float py, int width, int height, float r, float g, float b, float sizePx)
{
    // NDC = ((px+0.5)/W*2-1, (py+0.5)/H*2-1)
    const float scaleX =  2.0f / float(width);
    const float scaleY =  2.0f / float(height);
    const float offX   = -1.0f;
    const float offY   = -1.0f;

    const float p[1][5] = { { px + 0.5f, py + 0.5f, r, g, b } };
    drawPoints(&p[0][0], 1, scaleX, scaleY, offX, offY, sizePx);
}

// ---------- Mandelbrot/Fraktal: Bildschirm -> komplexe Ebene ----------
static inline std::pair<double,double>
screenToComplex(int px, int py, int width, int height, const RendererState& ctx, bool flipY)
{
    // Explicitly use ctx to prevent unused parameter warning
    [[maybe_unused]] const auto zoom = RS_ZOOM(ctx);
    [[maybe_unused]] const auto offsetX = RS_OFFSET_X(ctx);
    [[maybe_unused]] const auto offsetY = RS_OFFSET_Y(ctx);

    const double nx = ((double(px) + 0.5) / double(width))  * 2.0 - 1.0;
    double ny       = ((double(py) + 0.5) / double(height)) * 2.0 - 1.0;
    if (flipY) ny = -ny;

    const double aspect = double(width) / double(height);
    const double scale  = 1.0 / std::max(1e-12, RS_ZOOM(ctx));

    const double re = nx * aspect * scale + RS_OFFSET_X(ctx);
    const double im = ny * scale          + RS_OFFSET_Y(ctx);
    return {re, im};
}

// ---------- API ----------
void toggle(RendererState& ctx) {
#if defined(USE_HEATMAP_OVERLAY)
    ctx.heatmapOverlayEnabled = !ctx.heatmapOverlayEnabled;
#else
    (void)ctx;
#endif
}

void cleanup() {
    if (overlayVAO) glDeleteVertexArrays(1, &overlayVAO);
    if (overlayVBO) glDeleteBuffers(1, &overlayVBO);
    if (overlayShader) glDeleteProgram(overlayShader);
    if (pointVAO) glDeleteVertexArrays(1, &pointVAO);
    if (pointVBO) glDeleteBuffers(1, &pointVBO);
    if (pointProg) glDeleteProgram(pointProg);
    overlayVAO = overlayVBO = overlayShader = 0;
    pointVAO = pointVBO = pointProg = 0;
}

#include "heatmap_utils.hpp" // neu: für tileIndexToPixelCenter(...)

void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height,
                 int tileSize,
                 [[maybe_unused]] GLuint textureId,
                 [[maybe_unused]] RendererState& ctx)
{
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM] drawOverlay: entropy=%zu contrast=%zu enabled=%d size=%dx%d ts=%d",
                       entropy.size(), contrast.size(), ctx.heatmapOverlayEnabled ? 1 : 0,
                       width, height, tileSize);
#if !defined(RS_HAS_OFFSET) || !defined(RS_HAS_ZOOM)
        LUCHS_LOG_HOST("[HM] NOTE: Using default camera params (offset=(%.3f,%.3f), zoom=%.3f). "
                       "Define RS_HAS_OFFSET/RS_HAS_ZOOM and RS_*_EXPR to wire real fields.",
                       RS_OFFSET_X(ctx), RS_OFFSET_Y(ctx), RS_ZOOM(ctx));
#endif
    }
    if (!ctx.heatmapOverlayEnabled) return;

    static bool warned = false;
    if (entropy.empty() || contrast.empty()) {
        if (Settings::debugLogging && !warned) {
            LUCHS_LOG_HOST("[HM] WARN: entropy/contrast leer.");
            warned = true;
        }
        return;
    }

    const int tilesX = (width  + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int quadCount = tilesX * tilesY;

    // 1) Heatmap-Daten (Kachelraum) vorbereiten
    std::vector<float> data;
    data.reserve(quadCount * 6 * 3);

    float maxVal = 1e-6f;
    for (int i = 0; i < quadCount; ++i) {
        maxVal = std::max(maxVal, entropy[i] + contrast[i]);
    }

    int maxIdx = 0; float maxScore = -1.0f;
    for (int y = 0; y < tilesY; ++y) {
        for (int x = 0; x < tilesX; ++x) {
            const int idx = y * tilesX + x;
            const float raw = entropy[idx] + contrast[idx];
            const float v   = raw / maxVal;

            if (raw > maxScore) {
                maxScore = raw;
                maxIdx = idx;
            }

            const float px = float(x);
            const float py = float(y);
            const float quad[6][3] = {
                {px,     py,     v},
                {px + 1, py,     v},
                {px + 1, py + 1, v},
                {px,     py,     v},
                {px + 1, py + 1, v},
                {px,     py + 1, v}
            };
            for (auto& vert : quad) data.insert(data.end(), vert, vert + 3);
        }
    }

    // 2) Overlay-Pipeline
    if (overlayVAO == 0) {
        glGenVertexArrays(1, &overlayVAO);
        glGenBuffers(1, &overlayVBO);
        overlayShader = createOverlayProgram();
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[HM] Overlay init: VAO=%u VBO=%u Shader=%u", overlayVAO, overlayVBO, overlayShader);
        }
    }

    glUseProgram(overlayShader);

    // Mini-Overlay Größe/Position (Pixel) -> NDC
    constexpr int overlayPixelsX = 160, overlayPixelsY = 90, paddingX = 16, paddingY = 16;
    const float scaleX = (float(overlayPixelsX) / float(width)  / float(tilesX)) * 2.0f;
    const float scaleY = (float(overlayPixelsY) / float(height) / float(tilesY)) * 2.0f;
    const float offsetX = 1.0f - (float(overlayPixelsX + paddingX) / float(width)  * 2.0f);
    const float offsetY = 1.0f - (float(overlayPixelsY + paddingY) / float(height) * 2.0f);

    glUniform2f(glGetUniformLocation(overlayShader, "uScale"),  scaleX, scaleY);
    glUniform2f(glGetUniformLocation(overlayShader, "uOffset"), offsetX, offsetY);

    glBindVertexArray(overlayVAO);
    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));

    const GLenum errBefore = glGetError();
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(data.size() / 3));
    const GLenum errAfter  = glGetError();

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM] drawOverlay: verts=%zu  glErr=0x%x->0x%x", data.size()/3, errBefore, errAfter);
    }

    // 3) Self‑Check‑Punkte im Overlay (gleiche Transform)
    DrawHeatmapSelfCheck_OverlaySpace(tilesX, tilesY, scaleX, scaleY, offsetX, offsetY);

    // 4) Max‑Tile im Hauptbild markieren + Koordinaten-Logging (gemeinsame Hilfsfunktion)
    auto [centerPx, centerPy] = tileIndexToPixelCenter(maxIdx, tilesX, tilesY, width, height);
    DrawPoint_ScreenPixels(static_cast<float>(centerPx), static_cast<float>(centerPy),
                           width, height, 0.0f, 1.0f, 0.5f, 10.0f);

    if (Settings::debugLogging) {
        int bx = maxIdx % tilesX;
        int by = maxIdx / tilesX;
        const double ndcX = ((centerPx) / double(width)  - 0.5) * 2.0;
        const double ndcY = ((centerPy) / double(height) - 0.5) * 2.0;
        auto [reA, imA] = screenToComplex((int)std::floor(centerPx), (int)std::floor(centerPy),
                                          width, height, ctx, /*flipY=*/false);
        auto [reB, imB] = screenToComplex((int)std::floor(centerPx), (int)std::floor(centerPy),
                                          width, height, ctx, /*flipY=*/true);
        LUCHS_LOG_HOST("[HM] tiles=%dx%d ts=%d  maxIdx=%d -> (x=%d,y=%d)  centerPx=(%.1f,%.1f)  ndc=(%.5f, %.5f)",
                       tilesX, tilesY, tileSize, maxIdx, bx, by, centerPx, centerPy, ndcX, ndcY);
        LUCHS_LOG_HOST("[HM] complex(noFlip)= %.9f + i*%.9f   |   complex(Yflip)= %.9f + i*%.9f",
                       reA, imA, reB, imB);
        LUCHS_LOG_HOST("[HM] camera zoom=%.6f  offset=(%.9f, %.9f)  aspect=%.6f  (offset/zoom via adapter)",
                       RS_ZOOM(ctx), RS_OFFSET_X(ctx), RS_OFFSET_Y(ctx), double(width)/double(height));
    }

    // Aufräumen lokalen States
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
    glUseProgram(0);
}

} // namespace HeatmapOverlay


#pragma warning(pop) // ursprüngliche Warnungseinstellungen wiederherstellen
