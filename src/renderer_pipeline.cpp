///// Otter: MAUS header normalized; ASCII-only; no functional changes.
///// Schneefuchs: Header format per rules #60–62; path normalized.
///// Maus: Keep this as the only top header block; exact four lines.
///// Datei: src/renderer_pipeline.cpp
#include "pch.hpp"
#include "renderer_pipeline.hpp"
#include "renderer_resources.hpp"   // OpenGLUtils::updateTextureFromPBO(...)
#include "common.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include <GL/glew.h>
#include <cstdlib>

// ------------------------------- TU-Local State -------------------------------
namespace RendererPipeline {

namespace {
    static GLuint sProgram    = 0;
    static GLint  sUTex       = -1;
    static GLuint sDummyVAO   = 0;   // Core Profile verlangt ein gebundenes VAO
    static GLuint sTimeQuery  = 0;   // optionaler GPU-Timer

    // kleine State-Caches (nur in dieser TU)
    static GLuint s_lastProgram = 0;
    static GLuint s_lastTex2D   = 0;

    inline void bindProgram(GLuint p) {
        if (s_lastProgram != p) { glUseProgram(p); s_lastProgram = p; }
    }
    inline void bindTex2D(GLuint t) {
        if (s_lastTex2D != t)   { glBindTexture(GL_TEXTURE_2D, t); s_lastTex2D = t; }
    }

    // --- Mini Shader Utils (ASCII-Logs, deterministisch) ---------------------
    static GLuint compile(GLenum type, const char* src) {
        GLuint id = glCreateShader(type);
        glShaderSource(id, 1, &src, nullptr);
        glCompileShader(id);
        GLint ok = GL_FALSE;
        glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[2048]; GLsizei n = 0;
            glGetShaderInfoLog(id, (GLsizei)sizeof(log), &n, log);
            LUCHS_LOG_HOST("[GL][ERR] shader compile failed: %.*s", (int)n, log);
            glDeleteShader(id);
            return 0;
        }
        return id;
    }
    static GLuint link(GLuint vs, GLuint fs) {
        GLuint p = glCreateProgram();
        glAttachShader(p, vs);
        glAttachShader(p, fs);
        glLinkProgram(p);
        GLint ok = GL_FALSE;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048]; GLsizei n = 0;
            glGetProgramInfoLog(p, (GLsizei)sizeof(log), &n, log);
            LUCHS_LOG_HOST("[GL][ERR] program link failed: %.*s", (int)n, log);
            glDeleteProgram(p);
            return 0;
        }
        return p;
    }

    // --- Fullscreen-Triangle Shaders (gl_VertexID) ---------------------------
    static constexpr const char* VS = R"GLSL(#version 430 core
    out vec2 vUV;
    void main(){
        const vec2 pos[3] = vec2[3](
            vec2(-1.0, -1.0),
            vec2( 3.0, -1.0),
            vec2(-1.0,  3.0)
        );
        gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
        // grob [0,2] – clamped im FS; vermeidet Divisionen/Branches
        vUV = 0.5 * (pos[gl_VertexID] + 1.0);
    })GLSL";

    static constexpr const char* FS = R"GLSL(#version 430 core
    layout(location=0) out vec4 oColor;
    in vec2 vUV;
    uniform sampler2D uTex;
    void main(){
        vec2 uv = clamp(vUV, vec2(0.0), vec2(1.0));
        oColor = texture(uTex, uv);
    })GLSL";

    // einmalige Initialisierung (idempotent)
    static void ensurePipeline() {
        if (sProgram) return;

        GLuint vs = compile(GL_VERTEX_SHADER,   VS);
        GLuint fs = compile(GL_FRAGMENT_SHADER, FS);
        if (!vs || !fs) {
            LUCHS_LOG_HOST("[FATAL] shader build failed");
            std::exit(EXIT_FAILURE);
        }
        sProgram = link(vs, fs);
        glDeleteShader(vs);
        glDeleteShader(fs);
        if (!sProgram) {
            LUCHS_LOG_HOST("[FATAL] program link failed");
            std::exit(EXIT_FAILURE);
        }

        bindProgram(sProgram);
        sUTex = glGetUniformLocation(sProgram, "uTex");
        if (sUTex >= 0) glUniform1i(sUTex, 0);
        bindProgram(0);

        // Dummy-VAO fuer Core Profile
        glGenVertexArrays(1, &sDummyVAO);

        if constexpr (Settings::performanceLogging || Settings::debugLogging) {
            glGenQueries(1, &sTimeQuery);
            if constexpr (Settings::debugLogging)
                LUCHS_LOG_HOST("[GL] pipeline ready prog=%u vao=%u timerQ=%u", sProgram, sDummyVAO, sTimeQuery);
        }
    }
} // namespace

// ---------------------------------- API --------------------------------------

void init() {
    ensurePipeline();
    // Upload-Pfad nutzt RendererResources; PixelStore ist dort bereits definiert.
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPELINE] init done");
    }
}

void drawFullscreenQuad(GLuint tex) {
    ensurePipeline();

    // State sichern (minimal)
    GLint prevProg = 0; glGetIntegerv(GL_CURRENT_PROGRAM, &prevProg);
    GLint prevActiveTex = 0; glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTex);
    GLint prevTex0 = 0; glActiveTexture(GL_TEXTURE0); glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);

    GLboolean wasDepth = glIsEnabled(GL_DEPTH_TEST);
    GLboolean wasCull  = glIsEnabled(GL_CULL_FACE);
#ifdef GL_FRAMEBUFFER_SRGB
    GLboolean wasSRGB  = glIsEnabled(GL_FRAMEBUFFER_SRGB);
#endif

    if (wasDepth) glDisable(GL_DEPTH_TEST);
    if (wasCull)  glDisable(GL_CULL_FACE);
#ifdef GL_FRAMEBUFFER_SRGB
    if (wasSRGB)  glDisable(GL_FRAMEBUFFER_SRGB); // 1:1 Ausgabe
#endif

    // Draw
    bindProgram(sProgram);
    glActiveTexture(GL_TEXTURE0);
    bindTex2D(tex);
    glBindVertexArray(sDummyVAO);

    if constexpr (Settings::performanceLogging || Settings::debugLogging) {
        if (sTimeQuery) glBeginQuery(GL_TIME_ELAPSED, sTimeQuery);
    }

    glDrawArrays(GL_TRIANGLES, 0, 3);

    if constexpr (Settings::performanceLogging || Settings::debugLogging) {
        if (sTimeQuery) {
            glEndQuery(GL_TIME_ELAPSED);
            GLint ready = 0;
            glGetQueryObjectiv(sTimeQuery, GL_QUERY_RESULT_AVAILABLE, &ready);
            if (ready) {
                GLuint64 ns = 0;
                glGetQueryObjectui64v(sTimeQuery, GL_QUERY_RESULT, &ns);
                const double ms = double(ns) / 1.0e6;
                LUCHS_LOG_HOST("[TIME] FSQ gpu=%.3f ms", ms);
            }
        }
    }

    // State wiederherstellen
    glBindVertexArray(0);
    bindTex2D((GLuint)prevTex0);
    glActiveTexture((GLenum)prevActiveTex);
    bindProgram((GLuint)prevProg);

    if (wasCull)  glEnable(GL_CULL_FACE);
    if (wasDepth) glEnable(GL_DEPTH_TEST);
#ifdef GL_FRAMEBUFFER_SRGB
    if (wasSRGB)  glEnable(GL_FRAMEBUFFER_SRGB);
#endif
}

void cleanup() {
    if (sTimeQuery) { glDeleteQueries(1, &sTimeQuery); sTimeQuery = 0; }
    if (sDummyVAO)  { glDeleteVertexArrays(1, &sDummyVAO); sDummyVAO = 0; }
    if (sProgram)   { glDeleteProgram(sProgram); sProgram = 0; }

    sUTex = -1;
    s_lastProgram = 0;
    s_lastTex2D   = 0;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[CLEANUP] RendererPipeline resources cleaned up");
    }
}

} // namespace RendererPipeline
