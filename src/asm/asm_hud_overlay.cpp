///// Otter: ASM HUD overlay – draws tiny tilesX×tilesY Mandelbrot grid (ASM-based) as mini-panel, Panda-Größe wie Heatmap.
///// Schneefuchs: Eigenes GL-Programm + Texture; keine ASM-/CUDA-Aufrufe; State-Checks robust.
///// Maus: Rechts unten; einfache Warm-Colormap; ASCII-only Logs; no-op wenn Grid fehlt.
///// Datei: src/asm/asm_hud_overlay.cpp

#include "pch.hpp"
#include "asm_hud_overlay.hpp"

#include "renderer_state.hpp"
#include "frame_context.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

namespace {

// GL-Handles fuer das ASM-HUD-Panel
static GLuint sAsmHudVAO   = 0;
static GLuint sAsmHudVBO   = 0;
static GLuint sAsmHudProg  = 0;
static GLuint sAsmHudTex   = 0;
static GLint  uViewportPx  = -1;
static GLint  uGridTex     = -1;
static GLint  uAlpha       = -1;

// Textur-Groesse (Grid-Aufloesung)
static int sTexW = 0;
static int sTexH = 0;

// Simple Shader: quad in Pixelkoordinaten, Textur mit 0..1 Values -> Warm-Colormap
static const char* kAsmHudVS = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPositionPx;
layout(location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

uniform vec2 uViewportPx;

void main()
{
    vTexCoord = aTexCoord;

    // Pixel -> NDC: x: [0,w] -> [-1,+1], y: [0,h] (oben) -> +1..-1
    vec2 ndc;
    ndc.x = (aPositionPx.x / uViewportPx.x) * 2.0 - 1.0;
    ndc.y = 1.0 - (aPositionPx.y / uViewportPx.y) * 2.0;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
)GLSL";

static const char* kAsmHudFS = R"GLSL(
#version 430 core
in vec2 vTexCoord;
out vec4 outColor;

uniform sampler2D uGrid;
uniform float     uAlpha;

void main()
{
    float v = texture(uGrid, vTexCoord).r;
    v = clamp(v, 0.0, 1.0);

    // einfache Warm-Colormap: dunkel -> orange -> hell
    float r = v;
    float g = v * 0.6 + 0.2;
    float b = v * 0.3;

    outColor = vec4(r, g, b, uAlpha);
}
)GLSL";

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint id = glCreateShader(type);
    if (!id) return 0;
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    GLint ok = GL_FALSE;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        if constexpr (Settings::debugLogging) {
            char logBuf[512];
            GLsizei len = 0;
            glGetShaderInfoLog(id, (GLsizei)sizeof(logBuf), &len, logBuf);
            logBuf[(len >= 0 && len < (GLsizei)sizeof(logBuf)) ? len : (GLsizei)sizeof(logBuf) - 1] = '\0';
            LUCHS_LOG_HOST("[ASM/HUD] shader compile failed: %s", logBuf);
        }
        glDeleteShader(id);
        return 0;
    }
    return id;
}

static GLuint makeProgram()
{
    GLuint vs = compileShader(GL_VERTEX_SHADER,   kAsmHudVS);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, kAsmHudFS);
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return 0;
    }

    GLuint prog = glCreateProgram();
    if (!prog) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return 0;
    }

    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        if constexpr (Settings::debugLogging) {
            char logBuf[512];
            GLsizei len = 0;
            glGetProgramInfoLog(prog, (GLsizei)sizeof(logBuf), &len, logBuf);
            logBuf[(len >= 0 && len < (GLsizei)sizeof(logBuf)) ? len : (GLsizei)sizeof(logBuf) - 1] = '\0';
            LUCHS_LOG_HOST("[ASM/HUD] program link failed: %s", logBuf);
        }
        glDeleteProgram(prog);
        return 0;
    }

    return prog;
}

// Initialisiert VAO/VBO einmalig fuer ein Quad mit Position+TexCoord.
static void ensureVAO()
{
    if (sAsmHudVAO != 0 && sAsmHudVBO != 0) return;

    glGenVertexArrays(1, &sAsmHudVAO);
    glGenBuffers(1, &sAsmHudVBO);

    glBindVertexArray(sAsmHudVAO);
    glBindBuffer(GL_ARRAY_BUFFER, sAsmHudVBO);

    // layout: [pos.x,pos.y, tex.u,tex.v] als float[4]
    constexpr GLsizei stride = 4 * (GLsizei)sizeof(float);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * sizeof(float)));

    // keine Daten hier, werden zur Draw-Zeit per glBufferSubData gefuellt
}

// Sorgt fuer eine 2D-Textur mit tilesX×tilesY Float-Werten (R16F) und glTex(Sub)Image.
static void ensureTextureAndUpload(const float* data, int tilesX, int tilesY)
{
    if (!sAsmHudTex) {
        glGenTextures(1, &sAsmHudTex);
        glBindTexture(GL_TEXTURE_2D, sAsmHudTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        sTexW = 0;
        sTexH = 0;
    } else {
        glBindTexture(GL_TEXTURE_2D, sAsmHudTex);
    }

    GLint prevUnpack = 0;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevUnpack);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (sTexW != tilesX || sTexH != tilesY) {
        sTexW = tilesX;
        sTexH = tilesY;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, sTexW, sTexH, 0, GL_RED, GL_FLOAT, data);
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sTexW, sTexH, GL_RED, GL_FLOAT, data);
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, prevUnpack);
}

} // anon namespace

namespace asm_hud_overlay {

void draw(const RendererState& state, const FrameContext& ctx)
{
    // Master-Switch aus Settings: ohne Arbeit früh aussteigen
    if (!Settings::asmHudOverlayEnabled) {
        return;
    }

    const int tilesX = state.asmHudTilesX;
    const int tilesY = state.asmHudTilesY;
    if (tilesX <= 0 || tilesY <= 0) return;

    const size_t needed = static_cast<size_t>(tilesX) * static_cast<size_t>(tilesY);
    if (state.asmHudGrid.size() < needed) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ASM/HUD] grid size mismatch: have=%zu need=%zu",
                           state.asmHudGrid.size(), needed);
        }
        return;
    }

    if (ctx.width <= 0 || ctx.height <= 0) return;

    // Programm initialisieren
    if (!sAsmHudProg) {
        sAsmHudProg = makeProgram();
        if (!sAsmHudProg) {
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[ASM/HUD] program==0, skip draw");
            }
            return;
        }
        uViewportPx = glGetUniformLocation(sAsmHudProg, "uViewportPx");
        uGridTex    = glGetUniformLocation(sAsmHudProg, "uGrid");
        uAlpha      = glGetUniformLocation(sAsmHudProg, "uAlpha");
    }

    ensureVAO();

    // Grid normalisieren (0..1) fuer Textur-Upload – thread_local Buffer
    static thread_local std::vector<float> norm;
    norm.resize(needed);

    float vMin = state.asmHudGrid[0];
    float vMax = state.asmHudGrid[0];
    for (size_t i = 1; i < needed; ++i) {
        const float v = state.asmHudGrid[i];
        if (v < vMin) vMin = v;
        if (v > vMax) vMax = v;
    }

    const float range = (vMax > vMin) ? (vMax - vMin) : 1.0f;
    for (size_t i = 0; i < needed; ++i) {
        float v = (state.asmHudGrid[i] - vMin) / range;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        norm[i] = v;
    }

    // Perf-Log nur alle PerfLog::everyN Frames
    if constexpr (Settings::performanceLogging) {
        static int sLogCounter = 0;
        ++sLogCounter;
        if ((sLogCounter % Settings::PerfLog::everyN) == 0) {
            LUCHS_LOG_HOST("[ASM/HUD] tiles=%dx%d N=%zu vMin=%.4f vMax=%.4f",
                           tilesX, tilesY, needed, vMin, vMax);
        }
    }

    ensureTextureAndUpload(norm.data(), tilesX, tilesY);

    // Panel-Geometrie: rechts unten, Groesse in NDC aus Settings (Panda-Panels)
    const int viewW = ctx.width;
    const int viewH = ctx.height;

    const float panelWidthNdc  = Settings::hudPanelWidthNdc;
    const float panelHeightNdc = Settings::hudPanelHeightNdc;

    const int panelW = static_cast<int>(std::lround(0.5f * panelWidthNdc  * static_cast<float>(viewW)));
    const int panelH = static_cast<int>(std::lround(0.5f * panelHeightNdc * static_cast<float>(viewH)));

    constexpr int marginPx = 12;

    const int panelX1 = viewW - marginPx;
    const int panelX0 = panelX1 - panelW;
    const int panelY1 = viewH - marginPx;
    const int panelY0 = panelY1 - panelH;

    const int baseContentW = std::max(1, panelW);
    const int baseContentH = std::max(1, panelH);

    const float aspect = (tilesY > 0)
        ? static_cast<float>(tilesX) / static_cast<float>(tilesY)
        : 1.0f;

    int contentWPx = baseContentW;
    int contentHPx = baseContentH;

    if (aspect >= 1.0f) {
        contentWPx = baseContentW;
        contentHPx = std::max(1, static_cast<int>(std::l_
