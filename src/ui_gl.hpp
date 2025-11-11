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
    #error "OpenGL loader not found. Include pch.hpp (GLEW) before ui_gl.hpp, or make <GL/glew.h> / <glad/glad.h> available."
  #endif
#endif

#if __has_include("luchs_log_host.hpp")
  #include "luchs_log_host.hpp"
#endif

#include <cstdio>   // snprintf
#include <cstring>  // strlen

namespace UiGL {

// -------- Loader-/Extension-Utilities (GLEW/GLAD-agnostisch) ------------------
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

// -------- Safe-Label: keine Invalid-Handle-Meldungen mehr ---------------------
inline void safeObjectLabel(GLenum type, GLuint name, const char* label) {
#if defined(GL_KHR_debug)
    if (!hasKHRDebug() || name == 0) return;
    switch (type) {
        case GL_VERTEX_ARRAY: if (!glIsVertexArray(name)) return; break;
        case GL_BUFFER:       if (!glIsBuffer(name))       return; break;
        case GL_PROGRAM:      if (!glIsProgram(name))      return; break;
        case GL_SHADER:       if (!glIsShader(name))       return; break;
        default: /* other types: best-effort */ break;
    }
    const char* lab = (label && *label) ? label : "";
    // KHR_debug erlaubt -1 (nullterminiert)
    glObjectLabel(type, name, -1, lab);
#else
    (void)type; (void)name; (void)label;
#endif
}

// -------- Safe-Delete Helpers -------------------------------------------------
inline void safeDeleteVAO(GLuint& vao)   { if (vao && glIsVertexArray(vao)) glDeleteVertexArrays(1, &vao); vao = 0; }
inline void safeDeleteBuf(GLuint& buf)   { if (buf && glIsBuffer(buf))       glDeleteBuffers(1, &buf);     buf = 0; }
inline void safeDeletePrg(GLuint& prg)   { if (prg && glIsProgram(prg))      glDeleteProgram(prg);         prg = 0; }
inline void safeDeleteQry(GLuint& qry)   { if (qry) glDeleteQueries(1, &qry); qry = 0; } // glIsQuery optional

// -------- kompakte Shader/Program-Helfer -------------------------------------
inline GLuint makeShader(GLenum type, const char* src){
    GLuint sh = glCreateShader(type);
    if(!sh) return 0;
    safeObjectLabel(GL_SHADER, sh, (type==GL_VERTEX_SHADER)?"OTR_VS":(type==GL_FRAGMENT_SHADER)?"OTR_FS":"OTR_SH");

    glShaderSource(sh,1,&src,nullptr);
    glCompileShader(sh);
    GLint ok=0; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){
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
    GLuint v = makeShader(GL_VERTEX_SHADER,   vs); if(!v) return 0;
    GLuint f = makeShader(GL_FRAGMENT_SHADER, fs); if(!f){ glDeleteShader(v); return 0; }
    GLuint p = glCreateProgram(); if(!p){ glDeleteShader(v); glDeleteShader(f); return 0; }

    safeObjectLabel(GL_PROGRAM, p, "OTR_prog");
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

// -------- VAO/VBO-Layouts (State-Restore + Labels, keine Hotpath-Allokation) --
inline void ensurePanelVAO(GLuint& vao, GLuint& vbo){
    if (vao && glIsVertexArray(vao)) return;

    GLint prevVAO=0, prevBuf=0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevBuf);

    if (!vao || !glIsVertexArray(vao)) glGenVertexArrays(1,&vao);
    if (!vbo || !glIsBuffer(vbo))      glGenBuffers(1,&vbo);

    safeObjectLabel(GL_VERTEX_ARRAY, vao, "OTR_panelVAO");
    safeObjectLabel(GL_BUFFER,       vbo, "OTR_panelVBO");

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
    if (vao && glIsVertexArray(vao)) return;

    GLint prevVAO=0, prevBuf=0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevBuf);

    if (!vao || !glIsVertexArray(vao)) glGenVertexArrays(1,&vao);
    if (!vbo || !glIsBuffer(vbo))      glGenBuffers(1,&vbo);

    safeObjectLabel(GL_VERTEX_ARRAY, vao, "OTR_heatVAO");
    safeObjectLabel(GL_BUFFER,       vbo, "OTR_heatVBO");

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(float),(void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)prevBuf);
    glBindVertexArray((GLuint)prevVAO);
}

} // namespace UiGL
