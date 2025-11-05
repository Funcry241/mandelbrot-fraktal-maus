///// Otter: UI-GL-Helfer – kompakte Shader/Program-Utils, optionale KHR_debug-Labels, ASCII-Logs
///// Schneefuchs: /WX-safe; keine Seiteneffekte (State-Restore bei VAO/VBO); Loader optional nachladen
///// Maus: API = makeShader/makeProgram/ensurePanelVAO/ensureHeatVAO; keine Hotpath-Allokationen; deterministisch
///// Datei: src/ui_gl.hpp

#pragma once

// Dieser Header definiert nur kleine GL-Helfer. Er verlangt,
// dass der Übersetzungseinheit bereits ein GL-Loader bekannt ist.
// (bei euch: GLEW via pch.hpp). Zur Sicherheit versuchen wir,
// vorhandene Loader *optional* nachzuladen.

#if __has_include("pch.hpp")
  #include "pch.hpp"
#endif

// Wenn nach pch.hpp noch kein GL-Header drin ist, probieren wir erst GLEW, dann GLAD:
#if !defined(GLEW_VERSION) && !defined(GLAD_GL_H_)
  #if __has_include(<GL/glew.h>)
    #include <GL/glew.h>
  #elif __has_include(<glad/glad.h>)
    #include <glad/glad.h>
  #else
    // Fallback: Erkläre klar, was fehlt – aber nur wenn dieser Header
    // *ohne* vorherige GL-Header eingebunden wurde.
    #error "OpenGL loader not found. Include pch.hpp (GLEW) before ui_gl.hpp, or make <GL/glew.h> / <glad/glad.h> available."
  #endif
#endif

// Optionales Host-Logging ohne harte Abhängigkeit:
#if __has_include("luchs_log_host.hpp")
  #include "luchs_log_host.hpp"
#endif

#include <cstdio>   // snprintf

namespace UiGL {

// --- Loader-/Extension-Utilities (GLEW/GLAD-agnostisch) ----------------------
inline bool hasKHRDebug() {
#if defined(GL_KHR_debug)
  #if defined(GLEW_VERSION)
    return GLEW_KHR_debug != 0;
  #elif defined(GLAD_GL_H_)
    return glObjectLabel != nullptr;
  #else
    return true; // unbekannter Loader, aber KHR_debug sichtbar
  #endif
#else
  return false;
#endif
}

// kompakte Shader/Program-Helfer ------------------------------------------------

inline GLuint makeShader(GLenum type, const char* src){
    GLuint sh = glCreateShader(type); if(!sh) return 0;
#if defined(GL_KHR_debug)
    if (hasKHRDebug()) {
        const char* kind = (type==GL_VERTEX_SHADER?"VS":type==GL_FRAGMENT_SHADER?"FS":"SH");
        char label[64]; std::snprintf(label,sizeof(label),"OTR_%s", kind);
        glObjectLabel(GL_SHADER, sh, -1, label);
    }
#endif
    glShaderSource(sh,1,&src,nullptr);
    glCompileShader(sh);
    GLint ok=0; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){
        // kompakter Compile-Log (ASCII)
    #if defined(LUCHS_LOG_HOST)
        char log[1024] = {0};
        GLsizei n = 0;
        glGetShaderInfoLog(sh, (GLsizei)sizeof(log)-1, &n, log);
        LUCHS_LOG_HOST("[UI/GL][SH] compile fail type=0x%04X msg=%s", (unsigned)type, log);
    #endif
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

inline GLuint makeProgram(const char* vs, const char* fs){
    GLuint v = makeShader(GL_VERTEX_SHADER,vs); if(!v) return 0;
    GLuint f = makeShader(GL_FRAGMENT_SHADER,fs); if(!f){ glDeleteShader(v); return 0; }
    GLuint p = glCreateProgram(); if(!p){ glDeleteShader(v); glDeleteShader(f); return 0; }
#if defined(GL_KHR_debug)
    if (hasKHRDebug()) glObjectLabel(GL_PROGRAM, p, -1, "OTR_prog");
#endif
    glAttachShader(p,v); glAttachShader(p,f);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){
    #if defined(LUCHS_LOG_HOST)
        char log[1024] = {0};
        GLsizei n = 0;
        glGetProgramInfoLog(p, (GLsizei)sizeof(log)-1, &n, log);
        LUCHS_LOG_HOST("[UI/GL][PRG] link fail msg=%s", log);
    #endif
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

// VAO/VBO-Layouts einmalig konfigurieren (State-Restore + Labels) ----------------

inline void ensurePanelVAO(GLuint& vao, GLuint& vbo){
    if(vao) return;

    GLint prevVAO=0, prevBuf=0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevBuf);

    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
#if defined(GL_KHR_debug)
    if (hasKHRDebug()) {
        glObjectLabel(GL_VERTEX_ARRAY, vao, -1, "OTR_panelVAO");
        glObjectLabel(GL_BUFFER,       vbo, -1, "OTR_panelVBO");
    }
#endif
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(2*sizeof(float)));

    // Restore minimaler State
    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)prevBuf);
    glBindVertexArray((GLuint)prevVAO);
}

inline void ensureHeatVAO(GLuint& vao, GLuint& vbo){
    if(vao) return;

    GLint prevVAO=0, prevBuf=0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevBuf);

    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
#if defined(GL_KHR_debug)
    if (hasKHRDebug()) {
        glObjectLabel(GL_VERTEX_ARRAY, vao, -1, "OTR_heatVAO");
        glObjectLabel(GL_BUFFER,       vbo, -1, "OTR_heatVBO");
    }
#endif
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(float),(void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)prevBuf);
    glBindVertexArray((GLuint)prevVAO);
}

} // namespace UiGL
