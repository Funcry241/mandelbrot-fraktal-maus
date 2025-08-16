// Datei: src/renderer_pipeline.cpp
// 🐭 Maus-Kommentar: Kompakt, robust, Shader-Errors werden sauber erkannt. VAO-Handling und OpenGL-State sind clean - HUD/Heatmap bleiben garantiert sichtbar.
// 🦦 Otter: Keine OpenGL-Misere, Schneefuchs freut sich über stabile Pipelines. (Bezug zu Otter)
// 🐑 Schneefuchs: Fehlerquellen mit glGetError sichtbar gemacht, Upload deterministisch. (Bezug zu Schneefuchs)

#include "pch.hpp"
#include "renderer_pipeline.hpp"
#include "opengl_utils.hpp"
#include "common.hpp"
#include "settings.hpp"
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
        LUCHS_LOG_HOST("[FATAL] Shader program creation failed - aborting");
        std::exit(EXIT_FAILURE);
    }
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPELINE] Shader program created: %u", program);
    }

    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "uTex"), 0);
    glUseProgram(0);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPELINE] Uniform 'uTex' set to texture unit 0");
    }

    OpenGLUtils::createFullscreenQuad(&VAO, &VBO, &EBO);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPELINE] Fullscreen quad VAO=%u VBO=%u EBO=%u created", VAO, VBO, EBO);
    }
}

void updateTexture(GLuint pbo, GLuint tex, int width, int height) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Binding PBO=%u and Texture=%u for upload", pbo, tex);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Calling glTexSubImage2D with dimensions %dx%d", width, height);
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    if constexpr (Settings::debugLogging) {
        GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-UPLOAD] glTexSubImage2D glGetError() = 0x%04X", err);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Texture update from PBO complete");
    }
}

void drawFullscreenQuad(GLuint tex) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DRAW] About to draw fullscreen quad with Texture=%u", tex);
    }

    glUseProgram(program);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindVertexArray(VAO);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    if constexpr (Settings::debugLogging) {
        GLenum err = glGetError();
        LUCHS_LOG_HOST("[DRAW] glDrawElements glGetError() = 0x%04X", err);
    }

    glBindVertexArray(0);
    glUseProgram(0);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DRAW] Fullscreen quad drawn");
    }
}

void cleanup() {
    if (program) {
        glDeleteProgram(program);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CLEANUP] Deleted program %u", program);
        }
    }
    if (VAO) {
        glDeleteVertexArrays(1, &VAO);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CLEANUP] Deleted VAO %u", VAO);
        }
    }
    if (VBO) {
        glDeleteBuffers(1, &VBO);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CLEANUP] Deleted VBO %u", VBO);
        }
    }
    if (EBO) {
        glDeleteBuffers(1, &EBO);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CLEANUP] Deleted EBO %u", EBO);
        }
    }

    program = VAO = VBO = EBO = 0;
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CLEANUP] RendererPipeline resources cleaned up");
    }
}

} // namespace RendererPipeline
