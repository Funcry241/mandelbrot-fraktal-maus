///// Otter: MAUS header normalized; ASCII-only; no functional changes.
///// Schneefuchs: Header format per rules #60–62; path normalized.
///// Maus: Keep this as the only top header block; exact four lines.
///// Datei: src/renderer_pipeline.cpp

#include "pch.hpp"
#include "renderer_pipeline.hpp"
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
    static GLuint sTimeQuery  = 0;

    static GLuint s_lastProgram = 0;
    static GLuint s_lastTex2D   = 0;

    inline void bindProgram(GLuint p) {
        if (s_lastProgram != p) { glUseProgram(p); s_lastProgram = p; }
    }
    inline void bindTex2D(GLuint t) {
        if (s_lastTex2D != t)   { glBindTexture(GL_TEXTURE_2D, t); s_lastTex2D = t; }
    }

    // Track last active texture unit and fixed-function enables to avoid per-frame glGet*/glIsEnabled
    static GLenum    s_lastActiveTex = GL_TEXTURE0;
    static GLboolean s_depthEnabled  = GL_FALSE;
    static GLboolean s_cullEnabled   = GL_FALSE;
#ifdef GL_FRAMEBUFFER_SRGB
    static GLboolean s_srgbEnabled   = GL_FALSE;
#endif

    inline void bindActiveTex(GLenum unit) {
        if (s_lastActiveTex != unit) { glActiveTexture(unit); s_lastActiveTex = unit; }
    }
    inline void setDepthEnabled(bool on) {
        GLboolean want = on ? GL_TRUE : GL_FALSE;
        if (s_depthEnabled != want) { (on ? glEnable(GL_DEPTH_TEST) : glDisable(GL_DEPTH_TEST)); s_depthEnabled = want; }
    }
    inline void setCullEnabled(bool on) {
        GLboolean want = on ? GL_TRUE : GL_FALSE;
        if (s_cullEnabled != want) { (on ? glEnable(GL_CULL_FACE) : glDisable(GL_CULL_FACE)); s_cullEnabled = want; }
    }
#ifdef GL_FRAMEBUFFER_SRGB
    inline void setSRGBEnabled(bool on) {
        GLboolean want = on ? GL_TRUE : GL_FALSE;
        if (s_srgbEnabled != want) { (on ? glEnable(GL_FRAMEBUFFER_SRGB) : glDisable(GL_FRAMEBUFFER_SRGB)); s_srgbEnabled = want; }
    }
#endif

    // --- Mini Shader Utils (ASCII-Logs, deterministisch) ---------------------
    static GLuint compile(GLenum type, const char* src) {
        GLuint id = glCreateShader(type);
        glShaderSource(id, 1, &src, nullptr);
        glCompileShader(id);
        GLint ok = GL_FALSE;
        glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[2048] = {0};
            glGetShaderInfoLog(id, 2047, nullptr, log);
            LUCHS_LOG_HOST("[GL-SHADER][ERR] compile failed: %s", log);
            glDeleteShader(id);
            return 0;
        }
        return id;
    }

    static GLuint link(GLuint vs, GLuint fs) {
        GLuint prog = glCreateProgram();
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);
        GLint ok = GL_FALSE;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[2048] = {0};
            glGetProgramInfoLog(prog, 2047, nullptr, log);
            LUCHS_LOG_HOST("[GL-PROGRAM][ERR] link failed: %s", log);
            glDeleteProgram(prog);
            return 0;
        }
        return prog;
    }
} // namespace

// --- Simple FSQ Shader (GLSL 430 core) ---------------------------------------
static const char* VS = R"(#version 430 core
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID==1)?3.0:-1.0, (gl_VertexID==2)?3.0:-1.0);
    v_uv   = (p+1.0)*0.5;
    gl_Position = vec4(p,0,1);
}
)";

static const char* FS = R"(#version 430 core
layout(location=0) out vec4 o_color;
in vec2 v_uv;
uniform sampler2D uTex;
void main() {
    o_color = texture(uTex, v_uv);
}
)";

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

    // Seed tracked fixed-function states and texture unit/bindings (one-time)
    // Query once to initialize; later we track ourselves to avoid per-frame glGet*
    GLint activeTex = 0; 
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTex);
    s_lastActiveTex = (GLenum)activeTex;

    GLint boundTex2D = 0; 
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &boundTex2D);
    s_lastTex2D = (GLuint)boundTex2D;
    // restore previous active unit to avoid side effects
    glActiveTexture((GLenum)activeTex);

    s_depthEnabled = glIsEnabled(GL_DEPTH_TEST);
    s_cullEnabled  = glIsEnabled(GL_CULL_FACE);
#ifdef GL_FRAMEBUFFER_SRGB
    s_srgbEnabled  = glIsEnabled(GL_FRAMEBUFFER_SRGB);
#endif

    if constexpr (Settings::performanceLogging || Settings::debugLogging) {
        if (GLEW_VERSION_3_3 || GLEW_ARB_timer_query) {
            glGenQueries(1, &sTimeQuery);
        }
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPELINE] init done");
    }
}

void drawFullscreenQuad(GLuint tex) {
    ensurePipeline();

    // State sichern via TU-Tracker (keine per-frame glGet*)
    GLuint prevProg      = s_lastProgram;
    GLenum prevActiveTex = s_lastActiveTex;
    GLuint prevTex0      = s_lastTex2D;
    GLboolean prevDepth  = s_depthEnabled;
    GLboolean prevCull   = s_cullEnabled;
#ifdef GL_FRAMEBUFFER_SRGB
    GLboolean prevSRGB   = s_srgbEnabled;
#endif

    setDepthEnabled(false);
    setCullEnabled(false);
#ifdef GL_FRAMEBUFFER_SRGB
    setSRGBEnabled(true); // ensure linear->sRGB at FB
#endif

    // Draw
    bindProgram(sProgram);
    bindActiveTex(GL_TEXTURE0);
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

    // State wiederherstellen (ohne glGet*)
    glBindVertexArray(0);
    bindTex2D((GLuint)prevTex0);
    bindActiveTex((GLenum)prevActiveTex);
    bindProgram((GLuint)prevProg);
    setCullEnabled(prevCull == GL_TRUE);
    setDepthEnabled(prevDepth == GL_TRUE);
#ifdef GL_FRAMEBUFFER_SRGB
    setSRGBEnabled(prevSRGB == GL_TRUE);
#endif
}

// --- Thin wrappers to match header/API and keep headers & sources in sync -----
void init()    { ensurePipeline(); }
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
