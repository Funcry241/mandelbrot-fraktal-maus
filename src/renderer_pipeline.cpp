// Datei: src/renderer_pipeline.cpp
// Zeilen: 79
// 🐭 Maus-Kommentar: Sauber und ohne Altlast – `drawFullscreenQuad()` ist die einzige Renderfunktion. Shader lokal, VAO-Handling korrekt, kein `render()`-Legacy mehr. Schneefuchs: „So soll C++ schmecken.“

#include "pch.hpp"

#include "renderer_pipeline.hpp"
#include "opengl_utils.hpp"
#include "common.hpp"
#include <iostream>

namespace RendererPipeline {

static GLuint program = 0;
static GLuint VAO = 0, VBO = 0, EBO = 0;

static const char* vertexShaderSrc = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTex;
out vec2 vTex;
void main() {
    vTex = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

static const char* fragmentShaderSrc = R"GLSL(
#version 430 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vTex);
}
)GLSL";

void init() {
    program = OpenGLUtils::createProgramFromSource(vertexShaderSrc, fragmentShaderSrc);

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
