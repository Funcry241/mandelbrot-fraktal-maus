///// MAUS: Fullscreen pipeline — bind order, ASCII logs, optional GPU timer
// Datei: src/renderer_pipeline.cpp
// 🐭 Maus-Kommentar: Kompakt, robust, Shader-Errors werden sauber erkannt. VAO-Handling und OpenGL-State sind clean – HUD/Heatmap bleiben sichtbar.
// 🦦 Otter: Keine OpenGL-Misere, Schneefuchs freut sich über stabile Pipelines. (Bezug zu Otter)
// 🐑 Schneefuchs: Fehlerquellen mit glGetError sichtbar gemacht, Upload deterministisch. (Bezug zu Schneefuchs)
// 🐑 Schneefuchs: State-Change-Diät + optionale GPU-Timer-Query; Binds nur bei Änderung, ASCII-Logs.

#include "pch.hpp"
#include "renderer_pipeline.hpp"
#include "opengl_utils.hpp"
#include "common.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include <cstdlib>
#include <GL/glew.h>

namespace RendererPipeline {

static GLuint program = 0, VAO = 0, VBO = 0, EBO = 0;

// 🐑 Schneefuchs: lokale GL-State-Caches (nur in dieser TU).
namespace {
    static GLuint s_lastProgram  = 0;
    static GLuint s_lastVAO      = 0;
    static GLuint s_lastTex2D    = 0;
    static GLuint s_lastPBO      = 0;
    static GLint  s_unpackAlign  = -1; // -1 = unknown
    static GLint  s_unpackRowLen = -1;

    static GLuint s_timeQuery = 0;

    inline void bindProgram(GLuint p) { if (s_lastProgram != p) { glUseProgram(p); s_lastProgram = p; } }
    inline void bindVAO(GLuint vao)   { if (s_lastVAO != vao)   { glBindVertexArray(vao); s_lastVAO = vao; } }
    inline void bindTex2D(GLuint t)   { if (s_lastTex2D != t)   { glBindTexture(GL_TEXTURE_2D, t); s_lastTex2D = t; } }
    inline void bindPBO(GLuint pbo)   { if (s_lastPBO != pbo)   { glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo); s_lastPBO = pbo; } }
    inline void setUnpack(int a, int r){ if (s_unpackAlign!=a){glPixelStorei(GL_UNPACK_ALIGNMENT,a); s_unpackAlign=a;} if (s_unpackRowLen!=r){glPixelStorei(GL_UNPACK_ROW_LENGTH,r); s_unpackRowLen=r;} }
}

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
    // 🦊 Schneefuchs: idempotent – mehrfacher Aufruf erzeugt keine Ressourcen doppelt.
    if (program != 0 && VAO != 0) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPELINE] init skipped (already initialized)");
        }
        return;
    }

    program = OpenGLUtils::createProgramFromSource(vShader, fShader);
    if (!program) {
        LUCHS_LOG_HOST("[FATAL] Shader program creation failed - aborting");
        std::exit(EXIT_FAILURE);
    }
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPELINE] Shader program created: %u", program);
    }

    // Sampler auf Einheit 0 binden
    bindProgram(program);
    {
        const GLint loc = glGetUniformLocation(program, "uTex");
        if (loc >= 0) glUniform1i(loc, 0);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPELINE] Uniform 'uTex' set to texture unit 0 (loc=%d)", (int)loc);
        }
    }
    bindProgram(0);

    OpenGLUtils::createFullscreenQuad(&VAO, &VBO, &EBO);
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPELINE] Fullscreen quad VAO=%u VBO=%u EBO=%u created", VAO, VBO, EBO);
    }

    if constexpr (Settings::performanceLogging || Settings::debugLogging) {
        if (s_timeQuery == 0) {
            glGenQueries(1, &s_timeQuery);
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[PIPELINE] Created GL_TIME_ELAPSED query id=%u", s_timeQuery);
            }
        }
    }

    // Initiale PixelStore-Werte definieren (Upload erwartet 1/0)
    setUnpack(1, 0);
}

void updateTexture(GLuint pbo, GLuint tex, int width, int height) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Binding PBO=%u and Texture=%u for upload", pbo, tex);
    }

    setUnpack(1, 0);
    bindPBO(pbo);
    glActiveTexture(GL_TEXTURE0); // sicherheitshalber
    bindTex2D(tex);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] glTexSubImage2D %dx%d (PBO path)", width, height);
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    if constexpr (Settings::debugLogging) {
        const GLenum err = glGetError();
        LUCHS_LOG_HOST("[GL-UPLOAD] glTexSubImage2D glGetError() = 0x%04X", err);
    }

    // State sauber lassen
    bindTex2D(0);
    bindPBO(0);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[GL-UPLOAD] Texture update from PBO complete");
    }
}

void drawFullscreenQuad(GLuint tex) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DRAW] About to draw fullscreen quad with Texture=%u", tex);
    }

    bindProgram(program);

    // **sRGB deaktiviert** für klare 1:1-Ausgabe (Fenster ist sRGB-fähig)
    GLboolean srgbWas = GL_FALSE;
#ifdef GL_FRAMEBUFFER_SRGB
    srgbWas = glIsEnabled(GL_FRAMEBUFFER_SRGB);
    if (srgbWas) glDisable(GL_FRAMEBUFFER_SRGB);
#endif

    GLboolean wasDepth = glIsEnabled(GL_DEPTH_TEST);
    GLboolean wasCull  = glIsEnabled(GL_CULL_FACE);
    if (wasDepth) glDisable(GL_DEPTH_TEST);
    if (wasCull)  glDisable(GL_CULL_FACE);

    glActiveTexture(GL_TEXTURE0);
    bindTex2D(tex);
    bindVAO(VAO);

    // Debug: dunkler Clear, damit Draw sichtbar ist (nur im Debug)
    if constexpr (Settings::debugLogging) {
        glClearColor(0.05f, 0.06f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    if constexpr (Settings::performanceLogging || Settings::debugLogging) {
        if (s_timeQuery) {
            glBeginQuery(GL_TIME_ELAPSED, s_timeQuery);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            glEndQuery(GL_TIME_ELAPSED);

            GLint available = 0;
            glGetQueryObjectiv(s_timeQuery, GL_QUERY_RESULT_AVAILABLE, &available);
            if (available) {
                GLuint64 ns = 0;
                glGetQueryObjectui64v(s_timeQuery, GL_QUERY_RESULT, &ns);
                const double ms = (double)ns / 1.0e6;
                LUCHS_LOG_HOST("[TIME] FSQ gpu=%.3f ms", ms);
            }
        } else {
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }
    } else {
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    if constexpr (Settings::debugLogging) {
        const GLenum err = glGetError();
        LUCHS_LOG_HOST("[DRAW] glDrawElements glGetError() = 0x%04X", err);
    }

    // State wiederherstellen
    bindVAO(0);
    bindTex2D(0);
    bindProgram(0);
    if (wasCull)  glEnable(GL_CULL_FACE);
    if (wasDepth) glEnable(GL_DEPTH_TEST);
#ifdef GL_FRAMEBUFFER_SRGB
    if (srgbWas) glEnable(GL_FRAMEBUFFER_SRGB);
#endif

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DRAW] Fullscreen quad drawn");
    }
}

void cleanup() {
    if (s_timeQuery) {
        glDeleteQueries(1, &s_timeQuery);
        s_timeQuery = 0;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[CLEANUP] Deleted GL_TIME_ELAPSED query");
        }
    }
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
    s_lastProgram = s_lastVAO = s_lastTex2D = s_lastPBO = 0;
    s_unpackAlign = s_unpackRowLen = -1;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CLEANUP] RendererPipeline resources cleaned up");
    }
}

} // namespace RendererPipeline
