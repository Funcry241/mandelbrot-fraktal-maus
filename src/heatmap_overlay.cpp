// heatmap_overlay.cpp - Zeilen: 135

/*
Maus-Kommentar 🐭: Heatmap-Overlay mit OpenGL und GLSL – nun **oben rechts** statt unten rechts. Die Shader und Zeichnung bleiben unverändert, aber Offset/Scale wurden angepasst. Schneefuchs validiert so visuell die Aktivität im Fraktalbild, ohne ImGui.
*/

#include "pch.hpp"
#include "heatmap_overlay.hpp"
#include <algorithm>
#include <cmath>

namespace HeatmapOverlay {

static GLuint overlayVAO = 0;
static GLuint overlayVBO = 0;
static GLuint overlayShader = 0;
static bool showOverlay = true;

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
    return vec3(1.0 - v, v * 0.8, 0.2 + 0.8 * v);
}
void main() {
    FragColor = vec4(colormap(clamp(vValue, 0.0, 1.0)), 1.0);
}
)GLSL";

static GLuint compile(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    return shader;
}

static GLuint createShaderProgram() {
    GLuint vs = compile(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

void toggle() {
    showOverlay = !showOverlay;
}

void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height,
                 int tileSize,
                 GLuint textureId) {
    if (!showOverlay) return;

    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int quadCount = tilesX * tilesY;

    std::vector<float> data;
    data.reserve(quadCount * 6 * 3); // 6 vertices, each: x, y, val

    float maxVal = 1e-6f;
    for (int i = 0; i < quadCount; ++i) {
        maxVal = std::max(maxVal, entropy[i] + contrast[i]);
    }

    for (int y = 0; y < tilesY; ++y) {
        for (int x = 0; x < tilesX; ++x) {
            int idx = y * tilesX + x;
            float v = (entropy[idx] + contrast[idx]) / maxVal;

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

            for (auto& vertex : quad) {
                data.insert(data.end(), vertex, vertex + 3);
            }
        }
    }

    if (overlayVAO == 0) {
        glGenVertexArrays(1, &overlayVAO);
        glGenBuffers(1, &overlayVBO);
        overlayShader = createShaderProgram();
    }

    glUseProgram(overlayShader);

    const float scale = 4.0f;
    float overlayWidth = tilesX * scale;
    float overlayHeight = tilesY * scale;

    glUniform2f(glGetUniformLocation(overlayShader, "uScale"), scale * 2.0f / width, scale * 2.0f / height);
    glUniform2f(glGetUniformLocation(overlayShader, "uOffset"),
                1.0f - overlayWidth * 2.0f / width,
                1.0f - overlayHeight * 2.0f / height);

    glBindVertexArray(overlayVAO);
    glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));

    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(data.size() / 3));

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindVertexArray(0);
    glUseProgram(0);
}

} // namespace HeatmapOverlay
