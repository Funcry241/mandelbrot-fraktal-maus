// Datei: src/heatmap_overlay.cpp
// 🐭 Maus-Kommentar: Overlay-Zustand wird nicht mehr intern gespeichert. drawOverlay(ctx) steht jetzt global bereit,
// prüft ctx.overlayActive und ruft intern HeatmapOverlay::drawOverlay(...) auf. Damit ist die Integration in
// renderer_loop.cpp direkt möglich. Schneefuchs: Sichtbarkeit mit System.

#include "pch.hpp"
#include "heatmap_overlay.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "luchs_log_host.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace HeatmapOverlay {

static GLuint overlayVAO = 0;
static GLuint overlayVBO = 0;
static GLuint overlayShader = 0;

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

// --- Minimaler Selbsttest für Heatmap-Koordinaten (gleiche Transform wie Overlay) ---
static void DrawHeatmapSelfCheck(int tilesX, int tilesY, float scaleX, float scaleY, float offsetX, float offsetY)
{
    struct P { float x,y; float r,g,b; };
    const P pts[3] = {
        { 0.5f,             0.5f,              0.0f, 0.5f, 1.0f }, // blau links-unten
        { tilesX - 0.5f,    tilesY - 0.5f,     0.0f, 0.5f, 1.0f }, // blau rechts-oben
        { tilesX * 0.5f,    tilesY * 0.5f,     1.0f, 0.0f, 0.0f }, // rot Mitte
    };

    static const char* vs = R"GLSL(
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
    static const char* fs = R"GLSL(
        #version 430 core
        in vec3 vColor;
        out vec4 FragColor;
        void main(){ FragColor = vec4(vColor, 1.0); }
    )GLSL";

    static GLuint prog = 0, vao = 0, vbo = 0;
    if (!prog) {
        auto compile = [](GLenum type, const char* src){
            GLuint s = glCreateShader(type); glShaderSource(s,1,&src,nullptr); glCompileShader(s);
            GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
            if(!ok){ char log[1024]; glGetShaderInfoLog(s,1024,nullptr,log); LUCHS_LOG_HOST("[HM-SelfCheck] shader compile: %s", log); }
            return s;
        };
        GLuint v = compile(GL_VERTEX_SHADER,   vs);
        GLuint f = compile(GL_FRAGMENT_SHADER, fs);
        prog = glCreateProgram(); glAttachShader(prog,v); glAttachShader(prog,f); glLinkProgram(prog);
        GLint ok=0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if(!ok){ char log[1024]; glGetProgramInfoLog(prog,1024,nullptr,log); LUCHS_LOG_HOST("[HM-SelfCheck] link: %s", log); }
        glDeleteShader(v); glDeleteShader(f);
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(pts), pts, GL_DYNAMIC_DRAW);

    glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog,"uScale"),  scaleX, scaleY);
    glUniform2f(glGetUniformLocation(prog,"uOffset"), offsetX, offsetY);
    glUniform1f(glGetUniformLocation(prog,"uPointSize"), 12.0f);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(P), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(P), (void*)(2*sizeof(float)));

    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, 3);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
    glUseProgram(0);
}

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
        LUCHS_LOG_HOST("[SHADER ERROR] Linking failed: %s", log);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

void toggle(RendererState& ctx) {
    ctx.heatmapOverlayEnabled = !ctx.heatmapOverlayEnabled;
}

void cleanup() {
    if (overlayVAO) glDeleteVertexArrays(1, &overlayVAO);
    if (overlayVBO) glDeleteBuffers(1, &overlayVBO);
    if (overlayShader) glDeleteProgram(overlayShader);
    overlayVAO = overlayVBO = overlayShader = 0;
}

void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height,
                 int tileSize,
                 [[maybe_unused]] GLuint textureId,
                 RendererState& ctx) {
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM] drawOverlay called: entropy=%zu contrast=%zu enabled=%d",
                       entropy.size(), contrast.size(), ctx.heatmapOverlayEnabled ? 1 : 0);    
    }

    if (!ctx.heatmapOverlayEnabled) return;

    static bool warned = false;
    if (entropy.empty() || contrast.empty()) {
        if (Settings::debugLogging && !warned) {
            LUCHS_LOG_HOST("[DEBUG] HeatmapOverlay: entropy or contrast vector is empty.");
            warned = true;
        }
        return;
    }

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int quadCount = tilesX * tilesY;

    std::vector<float> data;
    data.reserve(quadCount * 6 * 3);

    // --- Zusatz-Log 2: Daten-Extrema sammeln ---
    float maxVal = 1e-6f;
    float minVal = std::numeric_limits<float>::max();
    for (int i = 0; i < quadCount; ++i) {
        float s = entropy[i] + contrast[i];
        maxVal = std::max(maxVal, s);
        minVal = std::min(minVal, s);
    }
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM-Data] Value range: min=%.6f max=%.6f", minVal, maxVal);
    }

    for (int y = 0; y < tilesY; ++y) {
        for (int x = 0; x < tilesX; ++x) {
            int idx = y * tilesX + x;
            float s = entropy[idx] + contrast[idx];

            // --- Zusatz-Log 3: Warnung bei NaN/negativen Werten (vor und nach Normierung) ---
            if (!std::isfinite(s) || s < 0.0f) {
                if (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[HM-WARN] Unusual raw value at tile (%d,%d) idx=%d: s=%.6f", x, y, idx, s);
                }
            }

            float v = s / maxVal;
            if (!std::isfinite(v) || v < 0.0f) {
                if (Settings::debugLogging) {
                    LUCHS_LOG_HOST("[HM-WARN] Unusual normalized value at tile (%d,%d) idx=%d: v=%.6f (s=%.6f, max=%.6f)",
                                   x, y, idx, v, s, maxVal);
                }
            }

            float px = static_cast<float>(x);
            float py = static_cast<float>(y);
            float quad[6][3] = {
                {px,     py,     v},
                {px + 1, py,     v},
                {px + 1, py + 1, v},
                {px,     py,     v},
                {px + 1, py + 1, v},
                {px,     py + 1, v}
            };
            for (auto& vertex : quad)
                data.insert(data.end(), vertex, vertex + 3);
        }
    }

    if (overlayVAO == 0) {
        glGenVertexArrays(1, &overlayVAO);
        glGenBuffers(1, &overlayVBO);
        overlayShader = createShaderProgram();
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[HM] Overlay initialized: VAO=%u VBO=%u Shader=%u", overlayVAO, overlayVBO, overlayShader);
        }
    }

    glUseProgram(overlayShader);

    // Transform für Einblendung unten rechts (kleines Overlay)
    constexpr int overlayPixelsX = 160, overlayPixelsY = 90, paddingX = 16, paddingY = 16;
    float scaleX = static_cast<float>(overlayPixelsX) / width / tilesX * 2.0f;
    float scaleY = static_cast<float>(overlayPixelsY) / height / tilesY * 2.0f;
    float offsetX = 1.0f - (static_cast<float>(overlayPixelsX + paddingX) / width * 2.0f);
    float offsetY = 1.0f - (static_cast<float>(overlayPixelsY + paddingY) / height * 2.0f);

    glUniform2f(glGetUniformLocation(overlayShader, "uScale"),  scaleX,  scaleY);
    glUniform2f(glGetUniformLocation(overlayShader, "uOffset"), offsetX, offsetY);

    // --- Zusatz-Log 1: NDC der Eck-/Mittel-Punkte (gleiche Transform wie Shader) ---
    if (Settings::debugLogging) {
        float blx = 0.5f * scaleX + offsetX;
        float bly = 0.5f * scaleY + offsetY;
        float trx = (tilesX - 0.5f) * scaleX + offsetX;
        float try_ = (tilesY - 0.5f) * scaleY + offsetY;
        float midx = (tilesX * 0.5f) * scaleX + offsetX;
        float midy = (tilesY * 0.5f) * scaleY + offsetY;
        LUCHS_LOG_HOST("[HM-Check] NDC BL=(%.3f, %.3f) TR=(%.3f, %.3f) MID=(%.3f, %.3f)",
                       blx, bly, trx, try_, midx, midy);
    }

    glBindVertexArray(overlayVAO);
    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));

    GLenum errBefore = glGetError();
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(data.size() / 3));
    GLenum errAfter = glGetError();

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM] drawOverlay: %zu vertices issued | glGetError = 0x%x -> 0x%x",
                       data.size() / 3, errBefore, errAfter);

        // Self‑Check: gleiche Transform, gleiche Ecke (blaue Punkte: BL/TR, roter Punkt: Mitte)
        DrawHeatmapSelfCheck(tilesX, tilesY, scaleX, scaleY, offsetX, offsetY);
    }

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
    glUseProgram(0);
}

} // namespace HeatmapOverlay
