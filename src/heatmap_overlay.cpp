///// Otter: Heatmap-Overlay - Diagnose & Mini-HUD; Shader 0 on error; cached uniforms; no state leaks.
///// Schneefuchs: Zustands-Restore (VAO/VBO/Program/Blend); ASCII-Logs; deterministisch; keine verdeckten Pfade.
///// Maus: Default ohne Marker/Points; Overlay y=0 unten; gleiche Datenquelle wie Zoom.
// Datei: src/heatmap_overlay.cpp

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

// 0 = aus, 1 = an.
#ifndef HEATMAP_DRAW_MAX_MARKER
#define HEATMAP_DRAW_MAX_MARKER 0
#endif
#ifndef HEATMAP_DRAW_SELF_CHECK
#define HEATMAP_DRAW_SELF_CHECK 0
#endif

// Quelle: RendererState – keine verdeckten Pfade.
// Nacktmull-Pullover: State nutzt center (double2).
#define RS_OFFSET_X(ctx) ((ctx).center.x)
#define RS_OFFSET_Y(ctx) ((ctx).center.y)
#define RS_ZOOM(ctx)     ((ctx).zoom)

namespace HeatmapOverlay {

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

static const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in float aValue;
out float vValue;
uniform vec2 uOffset;
uniform vec2 uScale;
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

static GLuint compile(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    if (!shader) {
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[SHADER ERROR] glCreateShader failed (type=%u)", (unsigned)type);
        return 0;
    }
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar log[2048] = {0};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[SHADER ERROR] Compilation failed (%s): %s",
                           type == GL_VERTEX_SHADER ? "Vertex" : "Fragment", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint linkProgram(GLuint vs, GLuint fs) {
    if (vs == 0 || fs == 0) return 0;
    GLuint prog = glCreateProgram();
    if (!prog) {
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[SHADER ERROR] glCreateProgram failed");
        return 0;
    }
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint success = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar log[2048] = {0};
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[SHADER ERROR] Linking failed: %s", log);
        glDeleteProgram(prog);
        prog = 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

static GLuint createOverlayProgram() {
    return linkProgram(compile(GL_VERTEX_SHADER,   vertexShaderSrc),
                       compile(GL_FRAGMENT_SHADER, fragmentShaderSrc));
}

#if HEATMAP_DRAW_MAX_MARKER || HEATMAP_DRAW_SELF_CHECK
static const char* pointVS = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec3 aColor;
uniform vec2 uScale;
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

static void ensurePointPipeline()
{
    if (pointProg) return;
    pointProg = linkProgram(compile(GL_VERTEX_SHADER, pointVS),
                            compile(GL_FRAGMENT_SHADER, pointFS));
    if (pointProg == 0) return;
    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);
    point_uScaleLoc  = glGetUniformLocation(pointProg, "uScale");
    point_uOffsetLoc = glGetUniformLocation(pointProg, "uOffset");
    point_uSizeLoc   = glGetUniformLocation(pointProg, "uPointSize");
}

static void drawPoints(const float* pts, int count, float scaleX, float scaleY, float offX, float offY, float pointSize)
{
    ensurePointPipeline();
    if (pointProg == 0) return;

    GLint prevVAO = 0, prevArray = 0, prevProg = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArray);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProg);

    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*5*count, nullptr, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*5*count, pts);

    glUseProgram(pointProg);
    glUniform2f(point_uScaleLoc,  scaleX, scaleY);
    glUniform2f(point_uOffsetLoc, offX, offY);
    glUniform1f(point_uSizeLoc,   pointSize);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(2*sizeof(float)));

    GLboolean wasBlend = GL_FALSE;
    glGetBooleanv(GL_BLEND, &wasBlend);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glDrawArrays(GL_POINTS, 0, count);

    if (!wasBlend) glDisable(GL_BLEND);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, prevArray);
    glBindVertexArray((GLuint)prevVAO);
    glUseProgram((GLuint)prevProg);
}

static void DrawHeatmapSelfCheck_OverlaySpace(int tilesX, int tilesY,
                                              float scaleX, float scaleY,
                                              float offsetX, float offsetY)
{
#if HEATMAP_DRAW_SELF_CHECK
    const float pts[5][5] = {
        { 0.5f,        0.5f,         0.0f, 0.0f, 1.0f },
        { tilesX-0.5f, 0.5f,         0.0f, 1.0f, 0.0f },
        { tilesX-0.5f, tilesY-0.5f,  1.0f, 1.0f, 0.0f },
        { 0.5f,        tilesY-0.5f,  1.0f, 0.0f, 1.0f },
        { tilesX*0.5f, tilesY*0.5f,  1.0f, 0.0f, 0.0f },
    };
    drawPoints(&pts[0][0], 5, scaleX, scaleY, offsetX, offsetY, 10.0f);
#else
    (void)tilesX; (void)tilesY; (void)scaleX; (void)scaleY; (void)offsetX; (void)offsetY;
#endif
}

static void DrawPoint_ScreenPixels(float px, float py, int width, int height, float r, float g, float b, float sizePx)
{
#if HEATMAP_DRAW_MAX_MARKER
    const float scaleX =  2.0f / float(width);
    const float scaleY =  2.0f / float(height);
    const float offX   = -1.0f;
    const float offY   = -1.0f;
    const float p[1][5] = { { px + 0.5f, py + 0.5f, r, g, b } };
    drawPoints(&p[0][0], 1, scaleX, scaleY, offX, offY, sizePx);
#else
    (void)px; (void)py; (void)width; (void)height; (void)r; (void)g; (void)b; (void)sizePx;
#endif
}
#endif // HEATMAP_DRAW_MAX_MARKER || HEATMAP_DRAW_SELF_CHECK

static inline std::pair<double,double>
screenToComplex(int px, int py, int width, int height, const RendererState& ctx, bool flipY)
{
    const double nx = ((double(px) + 0.5) / double(width))  * 2.0 - 1.0;
    double ny       = ((double(py) + 0.5) / double(height)) * 2.0 - 1.0;
    if (flipY) ny = -ny;

    const double aspect = double(width) / double(height);
    const double scale  = 1.0 / std::max(1e-12, (double)RS_ZOOM(ctx));

    const double re = nx * aspect * scale + (double)RS_OFFSET_X(ctx);
    const double im = ny * scale          + (double)RS_OFFSET_Y(ctx);
    return {re, im};
}

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
#if HEATMAP_DRAW_MAX_MARKER || HEATMAP_DRAW_SELF_CHECK
    if (pointVAO) glDeleteVertexArrays(1, &pointVAO);
    if (pointVBO) glDeleteBuffers(1, &pointVBO);
    if (pointProg) glDeleteProgram(pointProg);
    pointVAO = pointVBO = pointProg = 0;
    point_uScaleLoc = point_uOffsetLoc = point_uSizeLoc = -1;
#endif
    overlayVAO = overlayVBO = overlayShader = 0;
    overlay_uScaleLoc = overlay_uOffsetLoc = -1;
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

    const int tilesX = (width  + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int quadCount = tilesX * tilesY;

    std::vector<float> data;
    data.reserve(static_cast<size_t>(quadCount) * 6u * 3u);

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

            if (raw > maxScore) { maxScore = raw; maxIdx = idx; }

            const float px = float(x), py = float(y);
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

    if (overlayVAO == 0) {
        glGenVertexArrays(1, &overlayVAO);
        glGenBuffers(1, &overlayVBO);
        overlayShader = createOverlayProgram();
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

    constexpr int overlayHeightPx = 100;
    const float overlayAspect = float(tilesX) / float(tilesY);
    const int overlayWidthPx = static_cast<int>(overlayHeightPx * overlayAspect);
    constexpr int paddingX = 16, paddingY = 16;

    const float scaleX = (float(overlayWidthPx)  / float(width)  / float(tilesX)) * 2.0f;
    const float scaleY = (float(overlayHeightPx) / float(height) / float(tilesY)) * 2.0f;
    const float offsetX = 1.0f - (float(overlayWidthPx  + paddingX) / float(width)  * 2.0f);
    const float offsetY = 1.0f - (float(overlayHeightPx + paddingY) / float(height) * 2.0f);

    glUniform2f(overlay_uScaleLoc,  scaleX, scaleY);
    glUniform2f(overlay_uOffsetLoc, offsetX, offsetY);

    glBindVertexArray(overlayVAO);
    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(data.size() * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)(data.size() * sizeof(float)), data.data());

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));

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
    auto [centerPx, centerPy] = tileIndexToPixelCenter(maxIdx, tilesX, tilesY, width, height);
    DrawPoint_ScreenPixels(static_cast<float>(centerPx), static_cast<float>(centerPy),
                           width, height, 0.0f, 1.0f, 0.5f, 10.0f);
#endif

    if constexpr (Settings::debugLogging) {
        int bx = maxIdx % tilesX;
        int by = maxIdx / tilesX;
        auto [centerPxLog, centerPyLog] = tileIndexToPixelCenter(maxIdx, tilesX, tilesY, width, height);
        const double ndcX = ((centerPxLog) / double(width)  - 0.5) * 2.0;
        const double ndcY = ((centerPyLog) / double(height) - 0.5) * 2.0;
        auto [reA, imA] = screenToComplex((int)std::floor(centerPxLog), (int)std::floor(centerPyLog),
                                          width, height, ctx, /*flipY=*/false);
        auto [reB, imB] = screenToComplex((int)std::floor(centerPxLog), (int)std::floor(centerPyLog),
                                          width, height, ctx, /*flipY=*/true);
        LUCHS_LOG_HOST("[HM] tiles=%dx%d ts=%d  maxIdx=%d -> (x=%d,y=%d)  centerPx=(%.1f,%.1f)  ndc=(%.5f, %.5f)",
                       tilesX, tilesY, tileSize, maxIdx, bx, by, centerPxLog, centerPyLog, ndcX, ndcY);
        LUCHS_LOG_HOST("[HM] complex(noFlip)= %.9f + i*%.9f   |   complex(Yflip)= %.9f + i*%.9f",
                       reA, imA, reB, imB);
        LUCHS_LOG_HOST("[HM] camera zoom=%.6f  center=(%.9f, %.9f)  aspect=%.6f",
                       RS_ZOOM(ctx), RS_OFFSET_X(ctx), RS_OFFSET_Y(ctx), double(width)/double(height));
    }

    if (!wasBlend) glDisable(GL_BLEND);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, prevArray);
    glBindVertexArray((GLuint)prevVAO);
    glUseProgram((GLuint)prevProg);
}

} // namespace HeatmapOverlay

#pragma warning(pop)
