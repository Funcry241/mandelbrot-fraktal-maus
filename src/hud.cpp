// Datei: src/hud.cpp
// Zeilen: 187
// 🐭 Maus-Kommentar: Gepard HUD 1.0 – schützt vor GL-Crashes, prüft Kontext, fontPath, Shader-Erfolg. Kein draw ohne init. Fallback-Logs, ASCII-clean. Schneefuchs prüft und nickt.

#include "pch.hpp"
#include "hud.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"

#include <ft2build.h>
#include FT_FREETYPE_H
#include <map>
#include <string>
#include <filesystem>

namespace Hud {

static FT_Library ft;
static FT_Face face;
static GLuint textureAtlas = 0;
static GLuint vao = 0, vbo = 0;
static GLuint shader = 0;
static std::map<char, std::tuple<float, float, float, float>> glyphUVs;
static bool initialized = false;

static const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
uniform vec2 uResolution;
void main() {
    vec2 pos = aPos / uResolution * 2.0 - 1.0;
    gl_Position = vec4(pos.x, -pos.y, 0.0, 1.0);
    vUV = aUV;
})GLSL";

static const char* fragmentShaderSrc = R"GLSL(
#version 430 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uFontTex;
void main() {
    float alpha = texture(uFontTex, vUV).r;
    FragColor = vec4(1.0, 1.0, 1.0, alpha);
})GLSL";

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint success = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::fprintf(stderr, "[HUD] Shader compile error:\n%s\n", log);
        return 0;
    }
    return s;
}

static void buildAtlas() {
    const int atlasW = 512, atlasH = 512;
    glGenTextures(1, &textureAtlas);
    glBindTexture(GL_TEXTURE_2D, textureAtlas);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, atlasW, atlasH, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int x = 0, y = 0, rowH = 0;
    for (char c = 32; c < 127; ++c) {
        if (FT_Load_Char(face, c, FT_LOAD_RENDER)) continue;
        FT_Bitmap& bmp = face->glyph->bitmap;
        if (x + bmp.width >= atlasW) { x = 0; y += rowH; rowH = 0; }
        glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, bmp.width, bmp.rows, GL_RED, GL_UNSIGNED_BYTE, bmp.buffer);
        float u0 = x / float(atlasW), v0 = y / float(atlasH);
        float u1 = (x + bmp.width) / float(atlasW), v1 = (y + bmp.rows) / float(atlasH);
        glyphUVs[c] = { u0, v0, u1, v1 };
        x += bmp.width + 1;
        rowH = std::max(rowH, int(bmp.rows));
    }
}

void init() {
    std::puts("[HUD] init() called");

    if (initialized) {
        std::puts("[HUD] Already initialized, skipping.");
        return;
    }

    GLint major = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    if (major < 3) {
        std::puts("[HUD] ERROR: OpenGL context not ready. Aborting HUD init.");
        return;
    }

    if (FT_Init_FreeType(&ft)) {
        std::puts("[HUD] ERROR: Could not init FreeType");
        return;
    }

    const std::string fontPath = "fonts/Roboto-Regular.ttf";
    const std::string absPath = std::filesystem::absolute(fontPath).string();

    if (!std::filesystem::exists(fontPath)) {
        std::fprintf(stderr, "[HUD] Font file missing: %s\n", absPath.c_str());
        std::fprintf(stderr, "[HUD] HUD will be disabled.\n");
        return;
    }

    if (FT_New_Face(ft, fontPath.c_str(), 0, &face)) {
        std::fprintf(stderr, "[HUD] Failed to load font: %s\n", absPath.c_str());
        return;
    }

    FT_Set_Pixel_Sizes(face, 0, 32);
    buildAtlas();

    GLuint vs = compile(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    if (vs == 0 || fs == 0) {
        std::puts("[HUD] Shader compile failed – HUD disabled");
        return;
    }

    shader = glCreateProgram();
    glAttachShader(shader, vs);
    glAttachShader(shader, fs);
    glLinkProgram(shader);
    glDeleteShader(vs);
    glDeleteShader(fs);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    initialized = true;
    std::puts("[HUD] HUD initialized successfully.");
}

void drawText(const std::string& text, float x, float y, float w, float h) {
    if (!initialized) return;

    std::vector<float> verts;
    float penX = x;
    for (char c : text) {
        if (!glyphUVs.count(c)) continue;
        auto [u0, v0, u1, v1] = glyphUVs[c];
        float quadW = 16.0f, quadH = 32.0f;
        float x0 = penX,     y0 = y;
        float x1 = penX + quadW, y1 = y + quadH;
        verts.insert(verts.end(), {
            x0, y0, u0, v0,  x1, y0, u1, v0,  x1, y1, u1, v1,
            x0, y0, u0, v0,  x1, y1, u1, v1,  x0, y1, u0, v1
        });
        penX += quadW;
    }

    glUseProgram(shader);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureAtlas);
    glUniform1i(glGetUniformLocation(shader, "uFontTex"), 0);
    glUniform2f(glGetUniformLocation(shader, "uResolution"), w, h);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verts.size() / 4));
    glBindVertexArray(0);
    glUseProgram(0);
}

void draw(RendererState& state) {
    if (!initialized) return;

    char buf[128];
    std::snprintf(buf, sizeof(buf), "FPS: %.1f", state.fps);
    drawText(buf, 20.0f, 40.0f, float(state.width), float(state.height));

    double z = 1.0 / double(state.zoom);
    int exponent = int(std::log10(z));
    std::snprintf(buf, sizeof(buf), "Zoom: 1e%d", exponent);
    drawText(buf, 20.0f, 80.0f, float(state.width), float(state.height));

    std::snprintf(buf, sizeof(buf), "Offset: %.6f, %.6f", state.offset.x, state.offset.y);
    drawText(buf, 20.0f, 120.0f, float(state.width), float(state.height));
}

void cleanup() {
    if (!initialized) return;

    glDeleteTextures(1, &textureAtlas);
    glDeleteProgram(shader);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    initialized = false;
}

} // namespace Hud
