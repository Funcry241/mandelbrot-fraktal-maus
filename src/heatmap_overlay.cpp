///// Otter: Heatmap-Overlay – sanfte Zentralisierung (OpenGLUtils-Fallback), 0-Warnungen unter /W4 /WX, CUDA 13-sauber.
///// Schneefuchs: Keine Designänderung, nur minimale Ankopplung; Fallback bewahrt altes Verhalten; ASCII-Logs; deterministisch.
///// Maus: API fix: toggle()/cleanup() implementiert; createOverlayProgram nutzt OpenGLUtils, fällt sonst auf lokale compile/link zurück.
///// Datei: src/heatmap_overlay.cpp

#pragma warning(push)
#pragma warning(disable: 4100) // unused [[maybe_unused]] params in some builds

#include "pch.hpp"
#include "heatmap_overlay.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_utils.hpp"   // tileIndexToPixelCenter(...)
#include "opengl_utils.hpp"    // zentraler Shader-Helper (mit Fallback)
#include <algorithm>
#include <cmath>
#include <vector>
#include <array>
#include <cstdint>

#if !defined(GL_VERSION_4_3)
#  error "Requires OpenGL 4.3 core"
#endif

#ifndef HEATMAP_DRAW_MAX_MARKER
#define HEATMAP_DRAW_MAX_MARKER 0
#endif

#ifndef HEATMAP_DRAW_SELF_CHECK
#define HEATMAP_DRAW_SELF_CHECK 0
#endif

// Quelle: RendererState – keine verdeckten Pfade.
#define RS_OFFSET_X(ctx) ((ctx).center.x)
#define RS_OFFSET_Y(ctx) ((ctx).center.y)
#define RS_ZOOM(ctx)     ((ctx).zoom)

namespace HeatmapOverlay {

// ───────────────────────────── GL State ─────────────────────────────────────
static GLuint overlayVAO = 0;
static GLuint overlayVBO = 0;
static GLuint overlayShader = 0;
static GLint  overlay_uScaleLoc  = -1;
static GLint  overlay_uOffsetLoc = -1;

#if HEATMAP_DRAW_MAX_MARKER || HEATMAP_DRAW_SELF_CHECK
static GLuint pointProg   = 0;
static GLuint pointVAO    = 0;
static GLuint pointVBO    = 0;
static GLint  point_uScaleLoc   = -1;
static GLint  point_uOffsetLoc  = -1;
static GLint  point_uSizeLoc    = -1;
#endif

// ───────────────────────────── Shader Sources ────────────────────────────────
static const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in float aValue;

uniform vec2 uScale;   // von Kachel- in NDC-Skala
uniform vec2 uOffset;  // NDC-Offset

out float vValue;

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

// ───────────────────── Lokale Fallback-Shader-Helfer ────────────────────────
static GLuint compile(GLenum type, const char* src)
{
    GLuint sh = glCreateShader(type);
    if (!sh) return 0;
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = GL_FALSE;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 1 ? size_t(len) : size_t(1), 0);
        if (len > 1) glGetShaderInfoLog(sh, len, nullptr, log.data());
        LUCHS_LOG_HOST("[HM] ERROR: shader compile failed (type=%u) log='%s'", (unsigned)type, log.data());
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint linkProgram(GLuint vs, GLuint fs)
{
    if (!vs || !fs) { if (vs) glDeleteShader(vs); if (fs) glDeleteShader(fs); return 0; }
    GLuint prog = glCreateProgram();
    if (!prog) { glDeleteShader(vs); glDeleteShader(fs); return 0; }
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 1 ? size_t(len) : size_t(1), 0);
        if (len > 1) glGetProgramInfoLog(prog, len, nullptr, log.data());
        LUCHS_LOG_HOST("[HM] ERROR: program link failed log='%s'", log.data());
        glDeleteProgram(prog);
        prog = 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// Nur H7.a: Erst zentral versuchen, bei 0 auf lokalen Pfad zurückfallen.
// Keine weitere Logik/Seiteneffekte.
static GLuint createOverlayProgram()
{
    if (GLuint prog = OpenGLUtils::createProgramFromSource(vertexShaderSrc, fragmentShaderSrc))
        return prog;
    return linkProgram(compile(GL_VERTEX_SHADER, vertexShaderSrc),
                       compile(GL_FRAGMENT_SHADER, fragmentShaderSrc));
}

#if HEATMAP_DRAW_MAX_MARKER || HEATMAP_DRAW_SELF_CHECK
// ─────────────────────────── Punkt-Pipeline (opt.) ──────────────────────────
static const char* pointVS = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;
uniform vec2 uScale;
uniform vec2 uOffset;
uniform float uPointSize;
out vec3 vColor;
void main() {
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

static void ensurePointPipeline()
{
    if (pointProg) return;
    // Direkt zentral erzeugen (hier ohne Fallback – Marker sind optional)
    pointProg = OpenGLUtils::createProgramFromSource(pointVS, pointFS);
    if (pointProg == 0) return;
    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);
    point_uScaleLoc  = glGetUniformLocation(pointProg, "uScale");
    point_uOffsetLoc = glGetUniformLocation(pointProg, "uOffset");
    point_uSizeLoc   = glGetUniformLocation(pointProg, "uPointSize");
}

static void drawPoints(const float* pts, int count, float scaleX, float scaleY,
                       float offX, float offY, float pointSize)
{
    ensurePointPipeline();
    if (pointProg == 0) return;

    GLint prevVAO = 0, prevArray = 0, prevProg = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArray);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProg);

    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * size_t(count) * 5u, pts, GL_STREAM_DRAW);

    glUseProgram(pointProg);
    glUniform2f(point_uScaleLoc,  scaleX, scaleY);
    glUniform2f(point_uOffsetLoc, offX,  offY);
    glUniform1f(point_uSizeLoc, pointSize);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(uintptr_t)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(uintptr_t)(2 * sizeof(float)));

    glDrawArrays(GL_POINTS, 0, count);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)prevArray);
    glBindVertexArray((GLuint)prevVAO);
    glUseProgram((GLuint)prevProg);
}
#endif // HEATMAP_DRAW_MAX_MARKER || HEATMAP_DRAW_SELF_CHECK

// ───────────────────────────── Optional-Helper ──────────────────────────────
#if HEATMAP_DRAW_SELF_CHECK
#pragma warning(push)
#pragma warning(disable: 4505) // unreferenced static removed (wenn nicht genutzt)
static void DrawHeatmapSelfCheck_OverlaySpace(int tilesX, int tilesY,
                                              float scaleX, float scaleY,
                                              float offsetX, float offsetY)
{
    const float pts[5][5] = {
        { 0.5f,        0.5f,         0.0f, 0.0f, 1.0f },
        { tilesX-0.5f, 0.5f,         0.0f, 1.0f, 0.0f },
        { tilesX-0.5f, tilesY-0.5f,  1.0f, 1.0f, 0.0f },
        { 0.5f,        tilesY-0.5f,  1.0f, 0.0f, 1.0f },
        { tilesX*0.5f, tilesY*0.5f,  1.0f, 0.0f, 0.0f },
    };
    drawPoints(&pts[0][0], 5, scaleX, scaleY, offsetX, offsetY, 10.0f);
}
#pragma warning(pop)
#endif

#if HEATMAP_DRAW_MAX_MARKER
#pragma warning(push)
#pragma warning(disable: 4505)
static void DrawPoint_ScreenPixels(float px, float py, int width, int height,
                                   float r, float g, float b, float sizePx)
{
    const float scaleX =  2.0f / float(width);
    const float scaleY =  2.0f / float(height);
    const float offX   = -1.0f;
    const float offY   = -1.0f;

    const float pts[1][5] = { { px, py, r, g, b } };
    drawPoints(&pts[0][0], 1, scaleX, scaleY, offX, offY, sizePx);
}
#pragma warning(pop)
#endif

// ───────────────────────────── Lifetime / Draw / API ────────────────────────
static void releaseGLResources()
{
    if (overlayVAO) { glDeleteVertexArrays(1, &overlayVAO); overlayVAO = 0; }
    if (overlayVBO) { glDeleteBuffers(1, &overlayVBO); overlayVBO = 0; }
    if (overlayShader) { glDeleteProgram(overlayShader); overlayShader = 0; }
#if HEATMAP_DRAW_MAX_MARKER || HEATMAP_DRAW_SELF_CHECK
    if (pointVAO) { glDeleteVertexArrays(1, &pointVAO); pointVAO = 0; }
    if (pointVBO) { glDeleteBuffers(1, &pointVBO); pointVBO = 0; }
    if (pointProg) { glDeleteProgram(pointProg); pointProg = 0; }
    point_uScaleLoc = point_uOffsetLoc = point_uSizeLoc = -1;
#endif
    overlay_uScaleLoc = overlay_uOffsetLoc = -1;
}

void cleanup()   // API erwartet "cleanup()" (siehe Linkerfehlermeldung)
{
    releaseGLResources();
}

void shutdown()  // optional; Alias auf cleanup (falls woanders verwendet)
{
    releaseGLResources();
}

void toggle(RendererState& ctx) // API erwartet "toggle(RendererState&)"
{
    ctx.heatmapOverlayEnabled = !ctx.heatmapOverlayEnabled;
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM] overlay %s", ctx.heatmapOverlayEnabled ? "enabled" : "disabled");
    }
}

void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height,
                 int tileSize,
                 [[maybe_unused]] GLuint textureId,
                 [[maybe_unused]] RendererState& ctx)
{
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM] drawOverlay: entropy=%zu contrast=%zu enabled=%d size=%dx%d ts=%d",
                       entropy.size(), contrast.size(), ctx.heatmapOverlayEnabled ? 1 : 0,
                       width, height, tileSize);
    }
    if (!ctx.heatmapOverlayEnabled) return;
    if (entropy.empty() || contrast.empty()) {
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[HM] WARN: entropy/contrast empty, overlay skipped.");
        return;
    }

    // Dimensionen der Heatmap-Kacheln
    const int tilesX = std::max(1, width  / std::max(1, tileSize));
    const int tilesY = std::max(1, height / std::max(1, tileSize));
    const size_t numTiles = static_cast<size_t>(tilesX) * static_cast<size_t>(tilesY);
    if (entropy.size() < numTiles || contrast.size() < numTiles) {
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[HM] WARN: heatmap vectors too small: need=%zu gotE=%zu gotC=%zu",
                           numTiles, entropy.size(), contrast.size());
        return;
    }

    // Werte normieren (einfach, deterministisch)
    float minV = +1e30f, maxV = -1e30f;
    for (size_t i = 0; i < numTiles; ++i) {
        const float v = entropy[i];
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
    }
    const float span = (maxV - minV) > 1e-20f ? (maxV - minV) : 1.0f;

    // Finde Max-Index (nur Logging/Marker)
    size_t maxIdx = 0;
    float  maxVal = -1e30f;
    for (size_t i = 0; i < numTiles; ++i) {
        const float v = (entropy[i] - minV) / span;
        if (v > maxVal) { maxVal = v; maxIdx = i; }
    }

    // Vertexdaten für Dreiecks-Quads (pro Tile 6 Vertices, je (x,y,val))
    std::vector<float> data;
    data.reserve(6 * 3 * numTiles);
    const float scaleX = 2.0f / float(tilesX);
    const float scaleY = 2.0f / float(tilesY);
    const float offsetX = -1.0f;
    const float offsetY = -1.0f;

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            const size_t i = static_cast<size_t>(ty) * static_cast<size_t>(tilesX) + static_cast<size_t>(tx);
            const float v  = (entropy[i] - minV) / span;

            const float x0 = float(tx) + 0.0f;
            const float y0 = float(ty) + 0.0f;
            const float x1 = float(tx) + 1.0f;
            const float y1 = float(ty) + 1.0f;

            const float quad[6][3] = {
                { x0, y0, v }, { x1, y0, v }, { x1, y1, v },
                { x0, y0, v }, { x1, y1, v }, { x0, y1, v },
            };
            for (const auto& vert : quad) data.insert(data.end(), vert, vert + 3);
        }
    }

    if (overlayVAO == 0) {
        glGenVertexArrays(1, &overlayVAO);
        glGenBuffers(1, &overlayVBO);
        overlayShader = createOverlayProgram(); // H7.a
        if (overlayShader == 0) {
            if constexpr (Settings::debugLogging)
                LUCHS_LOG_HOST("[HM] ERROR: overlay shader creation failed.");
            return; // Shader==0 on error -> no draw, kein State-Leak
        }
        overlay_uScaleLoc  = glGetUniformLocation(overlayShader, "uScale");
        overlay_uOffsetLoc = glGetUniformLocation(overlayShader, "uOffset");
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[HM] Overlay init: VAO=%u VBO=%u Shader=%u", overlayVAO, overlayVBO, overlayShader);
        }
    }

    GLint prevVAO = 0, prevArray = 0, prevProg = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArray);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProg);
    GLboolean wasBlend = GL_FALSE;
    glGetBooleanv(GL_BLEND, &wasBlend);

    glUseProgram(overlayShader);

    glBindVertexArray(overlayVAO);
    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(), GL_STREAM_DRAW);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glUniform2f(overlay_uScaleLoc,  scaleX,  scaleY);
    glUniform2f(overlay_uOffsetLoc, offsetX, offsetY);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(uintptr_t)0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(uintptr_t)(2 * sizeof(float)));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const GLenum errBefore = glGetError();
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(data.size() / 3));
    const GLenum errAfter  = glGetError();

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM] drawOverlay: verts=%zu  glErr=0x%x->0x%x", data.size()/3, errBefore, errAfter);
    }

#if HEATMAP_DRAW_SELF_CHECK
    DrawHeatmapSelfCheck_OverlaySpace(tilesX, tilesY, scaleX, scaleY, offsetX, offsetY);
#endif

#if HEATMAP_DRAW_MAX_MARKER
    {
        // C4267 fix: index (size_t) -> int explicit cast (API erwartet int)
        auto [centerPx, centerPy] =
            tileIndexToPixelCenter(static_cast<int>(maxIdx), tilesX, tilesY, width, height);
        DrawPoint_ScreenPixels(static_cast<float>(centerPx), static_cast<float>(centerPy),
                               width, height, 0.0f, 1.0f, 0.5f, 10.0f);
    }
#endif

    if constexpr (Settings::debugLogging) {
        const int bx = static_cast<int>(maxIdx % static_cast<size_t>(tilesX));
        const int by = static_cast<int>(maxIdx / static_cast<size_t>(tilesX));
        auto [centerPxLog, centerPyLog] =
            tileIndexToPixelCenter(static_cast<int>(maxIdx), tilesX, tilesY, width, height); // C4267 safe
        const double ndcX = ((centerPxLog) / double(width)  - 0.5) * 2.0;
        const double ndcY = ((centerPyLog) / double(height) - 0.5) * 2.0;
        // Keine Abhängigkeit auf screenToComplex() an dieser Stelle – Null-Risiko.
        LUCHS_LOG_HOST("[HM] maxTile idx=%d (%d,%d) val=%.6f px=(%d,%d) ndc=(%.5f,%.5f) zoom=%.6e center=(%.9f, %.9f) ar=%.6f",
                       static_cast<int>(maxIdx), bx, by, maxVal,
                       static_cast<int>(centerPxLog), static_cast<int>(centerPyLog),
                       ndcX, ndcY,
                       RS_ZOOM(ctx), RS_OFFSET_X(ctx), RS_OFFSET_Y(ctx),
                       double(width)/double(height));
    }

    if (!wasBlend) glDisable(GL_BLEND);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)prevArray);
    glBindVertexArray((GLuint)prevVAO);
    glUseProgram((GLuint)prevProg);
}

} // namespace HeatmapOverlay

#pragma warning(pop)
