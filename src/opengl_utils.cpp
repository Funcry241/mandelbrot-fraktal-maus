///// Otter: OpenGL-Utils – Shader compile/link mit vollständigen Info-Logs; FSQ-Erzeugung mit sauberem State-Restore.
///// Schneefuchs: Deterministisch, ASCII-only; Debug-Gruppen optional; keine verdeckten Seiteneffekte.
///// Maus: Fehler beenden nicht; Funktionen liefern 0 bei Fehler und loggen klar (nur LUCHS_LOG_*).

#include "pch.hpp"
#include "opengl_utils.hpp"
#include "luchs_log_host.hpp"

#include <string>
#include <cstdint>

namespace OpenGLUtils {

// ───────────────────────────── helpers (TU-lokal) ────────────────────────────
namespace {
    inline void glDebugPush(const char* label) {
#if defined(GL_VERSION_4_3)
        if (glPushDebugGroup) glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, label);
#endif
    }
    inline void glDebugPop() {
#if defined(GL_VERSION_4_3)
        if (glPopDebugGroup) glPopDebugGroup();
#endif
    }
    inline void logGlError(const char* where) {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            LUCHS_LOG_HOST("[GL-ERROR] %s -> 0x%04X", where, err);
        }
    }

    // Schneefuchs: hole kompletten Info-Log (nicht nur 512 Zeichen)
    inline std::string getShaderInfoLog(GLuint shader) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        if (len <= 1) return {};
        std::string log;
        log.resize(static_cast<size_t>(len));
        GLsizei written = 0;
        glGetShaderInfoLog(shader, len, &written, log.data());
        if (written > 0 && written < len) log.resize(static_cast<size_t>(written));
        return log;
    }
    inline std::string getProgramInfoLog(GLuint prog) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        if (len <= 1) return {};
        std::string log;
        log.resize(static_cast<size_t>(len));
        GLsizei written = 0;
        glGetProgramInfoLog(prog, len, &written, log.data());
        if (written > 0 && written < len) log.resize(static_cast<size_t>(written));
        return log;
    }
} // namespace

// ─────────────────────────── shader compile/link ─────────────────────────────

// Schneefuchs: Kompiliere Shader; bei Fehler -> 0 zurück + vollständiges Log.
static GLuint compileShader(GLenum type, const char* src) {
    glDebugPush(type == GL_VERTEX_SHADER ? "compile vertex shader" : "compile fragment shader");

    if (!src || !*src) {
        LUCHS_LOG_HOST("[ShaderError] empty/null source (%s)", type == GL_VERTEX_SHADER ? "Vertex" : "Fragment");
        glDebugPop();
        return 0;
    }

    GLuint s = glCreateShader(type);
    if (!s) {
        LUCHS_LOG_HOST("[ShaderError] glCreateShader failed (type=%u)", (unsigned)type);
        glDebugPop();
        return 0;
    }

    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = GL_FALSE;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (ok != GL_TRUE) {
        const std::string log = getShaderInfoLog(s);
        LUCHS_LOG_HOST("[ShaderError] Compilation failed (%s): %s",
                       type == GL_VERTEX_SHADER ? "Vertex" : "Fragment",
                       log.empty() ? "(no info log)" : log.c_str());
        glDeleteShader(s);
        glDebugPop();
        return 0;
    }

    // Warnungen (Info-Log vorhanden, aber ok==true) trotzdem loggen
    const std::string warn = getShaderInfoLog(s);
    if (!warn.empty()) {
        LUCHS_LOG_HOST("[ShaderInfo] %s shader info: %s",
                       type == GL_VERTEX_SHADER ? "Vertex" : "Fragment",
                       warn.c_str());
    }

    logGlError("compileShader end");
    glDebugPop();
    return s;
}

GLuint createProgramFromSource(const char* vertexSrc, const char* fragmentSrc) {
    glDebugPush("link program");

    if (!vertexSrc || !*vertexSrc || !fragmentSrc || !*fragmentSrc) {
        LUCHS_LOG_HOST("[ShaderError] createProgramFromSource: null/empty source(s)");
        glDebugPop();
        return 0;
    }

    GLuint v = compileShader(GL_VERTEX_SHADER,  vertexSrc);
    if (v == 0) { glDebugPop(); return 0; }

    GLuint f = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
    if (f == 0) { glDeleteShader(v); glDebugPop(); return 0; }

    GLuint prog = glCreateProgram();
    if (!prog) {
        LUCHS_LOG_HOST("[ShaderError] glCreateProgram failed");
        glDeleteShader(v); glDeleteShader(f);
        glDebugPop();
        return 0;
    }

    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glLinkProgram(prog);

    GLint ok = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        const std::string log = getProgramInfoLog(prog);
        LUCHS_LOG_HOST("[ShaderError] Program link failed: %s",
                       log.empty() ? "(no info log)" : log.c_str());
        glDeleteShader(v);
        glDeleteShader(f);
        glDeleteProgram(prog);
        glDebugPop();
        return 0;
    }

    // Optional: Validate (ohne Crash, nur Log)
    glValidateProgram(prog);
    GLint validated = GL_FALSE;
    glGetProgramiv(prog, GL_VALIDATE_STATUS, &validated);
    if (validated != GL_TRUE) {
        const std::string vlog = getProgramInfoLog(prog);
        LUCHS_LOG_HOST("[ShaderInfo] Program validate status != GL_TRUE: %s",
                       vlog.empty() ? "(no info log)" : vlog.c_str());
    }

    // Detach & delete Shader-Objekte (Leak-Schutz)
    glDetachShader(prog, v);
    glDetachShader(prog, f);
    glDeleteShader(v);
    glDeleteShader(f);

    logGlError("createProgramFromSource end");
    glDebugPop();
    return prog;
}

// ───────────────────────────── fullscreen quad ───────────────────────────────

void createFullscreenQuad(GLuint* outVAO, GLuint* outVBO, GLuint* outEBO) {
    glDebugPush("create FSQ");

    if (!outVAO || !outVBO || !outEBO) {
        LUCHS_LOG_HOST("[GL-ERROR] createFullscreenQuad: null output pointer(s)");
        glDebugPop();
        return;
    }

    // Quad mit interleaved Position(2) + TexCoord(2)
    static constexpr float quad[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };
    static constexpr unsigned idx[] = { 0, 1, 2, 2, 3, 0 };

    // Vorherige Bindings sichern, damit wir keinen State hinterlassen.
    GLint prevVAO = 0, prevArray = 0, prevElem = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArray);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &prevElem);

    glGenVertexArrays(1, outVAO);
    glGenBuffers(1, outVBO);
    glGenBuffers(1, outEBO);

    if (*outVAO == 0 || *outVBO == 0 || *outEBO == 0) {
        LUCHS_LOG_HOST("[GL-ERROR] createFullscreenQuad: failed to generate buffers (VAO=%u VBO=%u EBO=%u)",
                       *outVAO, *outVBO, *outEBO);
        // Best-effort cleanup
        if (*outVAO) glDeleteVertexArrays(1, outVAO);
        if (*outVBO) glDeleteBuffers(1, outVBO);
        if (*outEBO) glDeleteBuffers(1, outEBO);
        *outVAO = *outVBO = *outEBO = 0;
        glDebugPop();
        return;
    }

    glBindVertexArray(*outVAO);

    glBindBuffer(GL_ARRAY_BUFFER, *outVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *outEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    constexpr GLsizei STRIDE = 4 * sizeof(float);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, STRIDE, (void*)(uintptr_t)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, STRIDE, (void*)(uintptr_t)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // VAO schließen und vorherigen State wiederherstellen
    glBindVertexArray(prevVAO);
    glBindBuffer(GL_ARRAY_BUFFER, prevArray);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElem);

    logGlError("createFullscreenQuad end");
    glDebugPop();
}

} // namespace OpenGLUtils
