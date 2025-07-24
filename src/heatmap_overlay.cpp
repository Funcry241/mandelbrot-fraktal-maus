// Datei: src/heatmap_overlay.cpp
// 🐭 Maus-Kommentar: Overlay-Zustand wird nicht mehr intern gespeichert. drawOverlay(ctx) steht jetzt global bereit, prüft ctx.overlayActive und ruft intern HeatmapOverlay::drawOverlay(...) auf. Damit ist die Integration in renderer_loop.cpp direkt möglich. Schneefuchs: Sichtbarkeit mit System.

#include "pch.hpp"
#include "heatmap_overlay.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include <algorithm>
#include <cmath>

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

static GLuint compile(GLenum type, const char* src) {
GLuint shader = glCreateShader(type);
glShaderSource(shader, 1, &src, nullptr);
glCompileShader(shader);
GLint success = 0;
glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
if (!success) {
GLchar log[1024];
glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
std::fprintf(stderr, "[SHADER ERROR] Compilation failed: %s\n", log);
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
std::fprintf(stderr, "[SHADER ERROR] Linking failed: %s\n", log);
}
glDeleteShader(vs);
glDeleteShader(fs);
return prog;
}

void toggle(RendererState& ctx) {
ctx.overlayEnabled = !ctx.overlayEnabled;
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
if (!ctx.overlayEnabled) return;

static bool warned = false;
if (entropy.empty() || contrast.empty()) {
    if (Settings::debugLogging && !warned) {
        std::fprintf(stderr, "[DEBUG] HeatmapOverlay: entropy or contrast vector is empty.\n");
        warned = true;
    }
    return;
}

const int tilesX = (width + tileSize - 1) / tileSize;
const int tilesY = (height + tileSize - 1) / tileSize;
const int quadCount = tilesX * tilesY;

std::vector<float> data;
data.reserve(quadCount * 6 * 3);

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
        for (auto& vertex : quad)
            data.insert(data.end(), vertex, vertex + 3);
    }
}

if (overlayVAO == 0) {
    glGenVertexArrays(1, &overlayVAO);
    glGenBuffers(1, &overlayVBO);
    overlayShader = createShaderProgram();
}

glUseProgram(overlayShader);

constexpr int overlayPixelsX = 160, overlayPixelsY = 90, paddingX = 16, paddingY = 16;
float scaleX = static_cast<float>(overlayPixelsX) / width / tilesX * 2.0f;
float scaleY = static_cast<float>(overlayPixelsY) / height / tilesY * 2.0f;
float offsetX = 1.0f - (static_cast<float>(overlayPixelsX + paddingX) / width * 2.0f);
float offsetY = 1.0f - (static_cast<float>(overlayPixelsY + paddingY) / height * 2.0f);

glUniform2f(glGetUniformLocation(overlayShader, "uScale"), scaleX, scaleY);
glUniform2f(glGetUniformLocation(overlayShader, "uOffset"), offsetX, offsetY);

glBindVertexArray(overlayVAO);
glBindBuffer(GL_ARRAY_BUFFER, overlayVBO);
glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);

glEnableVertexAttribArray(0);
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
glEnableVertexAttribArray(1);
glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));

glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(data.size() / 3));

GLenum err = glGetError();
if (Settings::debugLogging && err != GL_NO_ERROR) {
    std::fprintf(stderr, "[DEBUG] HeatmapOverlay: OpenGL error 0x%x\n", err);
}

glDisableVertexAttribArray(0);
glDisableVertexAttribArray(1);
glBindVertexArray(0);
glUseProgram(0);

}

} // namespace HeatmapOverlay

// 📣 Globale Funktion entfernt – benutze explizit HeatmapOverlay::drawOverlay im Render-Loop.
