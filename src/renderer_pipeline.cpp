// Datei: src/renderer_pipeline.cpp
// 🐭 Maus-Kommentar: Kompakt, robust, Shader-Errors werden sauber erkannt. VAO-Handling und OpenGL-State sind clean - HUD/Heatmap bleiben garantiert sichtbar. Otter: Keine OpenGL-Misere, Schneefuchs freut sich über stabile Pipelines.

#include "pch.hpp"
#include "renderer_pipeline.hpp"
#include "opengl_utils.hpp"
#include "common.hpp"
#include "luchs_log_host.hpp"
#include <cstdlib>

namespace RendererPipeline {

static GLuint program = 0, VAO = 0, VBO = 0, EBO = 0;

static constexpr const char* vShader = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTex;
out vec2 vTex;
void main() { vTex = aTex; gl_Position = vec4(aPos, 0.0, 1.0); }
)GLSL";

static constexpr const char* fShader = R"GLSL(
#version 430 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uTex;
void main() { FragColor = texture(uTex, vTex); }
)GLSL";

void init() {
    program = OpenGLUtils::createProgramFromSource(vShader, fShader);
    if (!program) {
        LUCHS_LOG_HOST("[FATAL] Shaderprogramm konnte nicht erstellt werden - OpenGL-Abbruch");
        std::exit(EXIT_FAILURE);
    }

    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "uTex"), 0);
    glUseProgram(0);

    OpenGLUtils::createFullscreenQuad(&VAO, &VBO, &EBO);
}

void updateTexture(GLuint pbo, GLuint tex, int width, int height) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void drawFullscreenQuad(GLuint tex) {
    glUseProgram(program);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    // Kein Blend-Disable - HUD benötigt Alpha-Blending!
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

void cleanup() {
    if (program) glDeleteProgram(program);
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (EBO) glDeleteBuffers(1, &EBO);
    program = VAO = VBO = EBO = 0;
}

} // namespace RendererPipeline
