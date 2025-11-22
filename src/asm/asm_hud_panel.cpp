///// Otter: ASM HUD panel – bottom-right ASM mini-fractal; grayscale heat with soft gold.
///// Schneefuchs: Eigenes GL-Panel (VAO/VBO/EBO/Prog/Tex); keine Nebenwirkungen auf CUDA/Heatmap.
///// Maus: Data: RendererState::asmHudGrid/TilesX/TilesY; NDC aus Viewport; ASCII-only, kein Logging-Zwang.
///// Datei: src/asm/asm_hud_panel.cpp
#include "pch.hpp"

#include "asm/asm_hud_panel.hpp"
#include "renderer_state.hpp"
#include "ui_gl.hpp"

#include <algorithm>
#include <vector>
#include <cmath>

namespace
{
    // Panel-Layout (als Bruchteil des Viewports)
    constexpr float PANEL_WIDTH_FRACTION  = 0.30f;
    constexpr float PANEL_HEIGHT_FRACTION = 0.30f;
    constexpr float PANEL_MARGIN_PX       = 16.0f;

    // GL-Handles
    static GLuint sVao = 0;
    static GLuint sVbo = 0;
    static GLuint sEbo = 0;
    static GLuint sProg = 0;
    static GLuint sTex = 0;

    // Uniform-Locations
    static GLint sLocResolution = -1;
    static GLint sLocPanelRect  = -1;
    static GLint sLocGridSize   = -1;
    static GLint sLocValueScale = -1;
    static GLint sLocValueBias  = -1;
    static GLint sLocSampler    = -1;

    // Aktuelle Texturgröße (Tiles)
    static int sTexWidth = 0;
    static int sTexHeight = 0;

    // GLSL: einfacher Panel-Shader mit Grid-Sampling
    static const char* kAsmHudVS = R"GLSL(
        #version 430 core
        layout(location = 0) in vec2 aLocalPos; // (0..1, 0..1) Panelraum
        layout(location = 1) in vec2 aUV;       // (0..1, 0..1) UV-Raum

        out vec2 vUV;

        uniform vec2 uResolution; // Viewport (px)
        uniform vec4 uPanelRect;  // x, y, w, h (px), y von unten

        void main()
        {
            vUV = aUV;

            // Panel-Position in Pixeln
            vec2 panelPx = uPanelRect.xy + aLocalPos * uPanelRect.zw;

            // Pixel → NDC (-1..+1), y=0 unten
            vec2 ndc = vec2(
                (panelPx.x / uResolution.x) * 2.0 - 1.0,
                (panelPx.y / uResolution.y) * 2.0 - 1.0
            );

            gl_Position = vec4(ndc, 0.0, 1.0);
        }
    )GLSL";

    static const char* kAsmHudFS = R"GLSL(
        #version 430 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D uGridTex;
        uniform vec2  uGridSize;   // tilesX, tilesY
        uniform float uValueScale; // 1 / (max - min) oder 1
        uniform float uValueBias;  // -min * scale, oder 0

        void main()
        {
            // Grid-Sampling: vUV (0..1) → Zelle (i,j) → Zentrum
            vec2 gridSize = max(uGridSize, vec2(1.0));
            vec2 cell = floor(vUV * gridSize);
            vec2 uvCell = (cell + vec2(0.5)) / gridSize;

            float raw = texture(uGridTex, uvCell).r;
            float v = raw * uValueScale + uValueBias;
            v = clamp(v, 0.0, 1.0);

            // leichte Gamma-Korrektur
            v = sqrt(v);

            // einfache Blau-Gold-Farbskala
            vec3 colLow  = vec3(0.0, 0.05, 0.20);
            vec3 colHigh = vec3(1.0, 0.85, 0.40);
            vec3 col = mix(colLow, colHigh, v);

            float alpha = 0.90;
            FragColor = vec4(col, alpha);
        }
    )GLSL";

    bool ensureProgram()
    {
        if (sProg != 0)
            return true;

        sProg = UiGL::makeProgram(kAsmHudVS, kAsmHudFS);
        if (!sProg)
            return false;

        sLocResolution = glGetUniformLocation(sProg, "uResolution");
        sLocPanelRect  = glGetUniformLocation(sProg, "uPanelRect");
        sLocGridSize   = glGetUniformLocation(sProg, "uGridSize");
        sLocValueScale = glGetUniformLocation(sProg, "uValueScale");
        sLocValueBias  = glGetUniformLocation(sProg, "uValueBias");
        sLocSampler    = glGetUniformLocation(sProg, "uGridTex");

        return true;
    }

    bool ensureGeometry()
    {
        if (sVao != 0)
            return true;

        // Quad im lokalen Panelraum (0..1) + UV
        const float verts[] = {
            // aLocalPos.x, aLocalPos.y, aUV.x, aUV.y
            0.0f, 0.0f, 0.0f, 0.0f, // bottom-left
            1.0f, 0.0f, 1.0f, 0.0f, // bottom-right
            1.0f, 1.0f, 1.0f, 1.0f, // top-right
            0.0f, 1.0f, 0.0f, 1.0f  // top-left
        };

        const unsigned indices[] = {
            0u, 1u, 2u,
            0u, 2u, 3u
        };

        glGenVertexArrays(1, &sVao);
        glBindVertexArray(sVao);

        glGenBuffers(1, &sVbo);
        glBindBuffer(GL_ARRAY_BUFFER, sVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

        glGenBuffers(1, &sEbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sEbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        const GLsizei stride = 4 * static_cast<GLsizei>(sizeof(float));

        // aLocalPos (0,1)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(
            0,
            2,
            GL_FLOAT,
            GL_FALSE,
            stride,
            reinterpret_cast<void*>(0)
        );

        // aUV (2,3)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(
            1,
            2,
            GL_FLOAT,
            GL_FALSE,
            stride,
            reinterpret_cast<void*>(2 * sizeof(float))
        );

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        return true;
    }

    bool ensureTexture()
    {
        if (sTex != 0)
            return true;

        glGenTextures(1, &sTex);
        glBindTexture(GL_TEXTURE_2D, sTex);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Placeholder-Textur (1x1) – wird beim ersten Draw ersetzt
        const float zero = 0.0f;
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R32F,
            1,
            1,
            0,
            GL_RED,
            GL_FLOAT,
            &zero
        );

        glBindTexture(GL_TEXTURE_2D, 0);
        sTexWidth = 1;
        sTexHeight = 1;
        return true;
    }

    void uploadGridToTexture(const RendererState& state)
    {
        const int tilesX = state.asmHudTilesX;
        const int tilesY = state.asmHudTilesY;

        if (tilesX <= 0 || tilesY <= 0)
            return;

        const std::size_t expected = static_cast<std::size_t>(tilesX) * static_cast<std::size_t>(tilesY);
        if (state.asmHudGrid.size() != expected)
            return;

        if (!ensureTexture())
            return;

        glBindTexture(GL_TEXTURE_2D, sTex);

        // Textur neu allokieren, falls die Größe sich geändert hat
        if (tilesX != sTexWidth || tilesY != sTexHeight)
        {
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_R32F,
                tilesX,
                tilesY,
                0,
                GL_RED,
                GL_FLOAT,
                state.asmHudGrid.data()
            );
            sTexWidth = tilesX;
            sTexHeight = tilesY;
        }
        else
        {
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                tilesX,
                tilesY,
                GL_RED,
                GL_FLOAT,
                state.asmHudGrid.data()
            );
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void computeMinMax(const RendererState& state, float& outMin, float& outMax)
    {
        const int tilesX = state.asmHudTilesX;
        const int tilesY = state.asmHudTilesY;

        const std::size_t expected = static_cast<std::size_t>(tilesX) * static_cast<std::size_t>(tilesY);
        if (tilesX <= 0 || tilesY <= 0 || state.asmHudGrid.size() != expected)
        {
            outMin = 0.0f;
            outMax = 1.0f;
            return;
        }

        auto mm = std::minmax_element(state.asmHudGrid.begin(), state.asmHudGrid.end());
        outMin = *mm.first;
        outMax = *mm.second;

        if (!(outMax > outMin))
        {
            outMin = 0.0f;
            outMax = 1.0f;
        }
    }
} // anonymous namespace

namespace AsmHudPanel
{

void draw(const RendererState& state, int viewportWidth, int viewportHeight)
{
    const int tilesX = state.asmHudTilesX;
    const int tilesY = state.asmHudTilesY;

    if (viewportWidth <= 0 || viewportHeight <= 0)
        return;

    if (tilesX <= 0 || tilesY <= 0)
        return;

    const std::size_t expected = static_cast<std::size_t>(tilesX) * static_cast<std::size_t>(tilesY);
    if (state.asmHudGrid.size() != expected)
        return;

    if (!ensureProgram())
        return;
    if (!ensureGeometry())
        return;
    if (!ensureTexture())
        return;

    // Daten in Textur laden
    uploadGridToTexture(state);

    // Min/Max für Normalisierung
    float vMin = 0.0f;
    float vMax = 1.0f;
    computeMinMax(state, vMin, vMax);

    float scale = 1.0f;
    float bias = 0.0f;
    if (vMax > vMin)
    {
        scale = 1.0f / (vMax - vMin);
        bias = -vMin * scale;
    }

    // Panel-Rechteck im Viewport (px)
    const float vpW = static_cast<float>(viewportWidth);
    const float vpH = static_cast<float>(viewportHeight);

    const float panelW = vpW * PANEL_WIDTH_FRACTION;
    const float panelH = vpH * PANEL_HEIGHT_FRACTION;

    const float x = vpW - panelW - PANEL_MARGIN_PX;
    const float y = PANEL_MARGIN_PX; // von unten

    const float panelRect[4] = { x, y, panelW, panelH };

    glUseProgram(sProg);

    if (sLocResolution >= 0)
        glUniform2f(sLocResolution, vpW, vpH);
    if (sLocPanelRect >= 0)
        glUniform4f(sLocPanelRect, panelRect[0], panelRect[1], panelRect[2], panelRect[3]);
    if (sLocGridSize >= 0)
        glUniform2f(sLocGridSize, static_cast<float>(tilesX), static_cast<float>(tilesY));
    if (sLocValueScale >= 0)
        glUniform1f(sLocValueScale, scale);
    if (sLocValueBias >= 0)
        glUniform1f(sLocValueBias, bias);
    if (sLocSampler >= 0)
        glUniform1i(sLocSampler, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sTex);

    glBindVertexArray(sVao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

} // namespace AsmHudPanel
