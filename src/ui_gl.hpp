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

namespace UiGL {

// kompakte Shader/Program-Helfer
inline GLuint makeShader(GLenum type, const char* src){
    GLuint sh = glCreateShader(type); if(!sh) return 0;
    glShaderSource(sh,1,&src,nullptr); glCompileShader(sh);
    GLint ok=0; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){ glDeleteShader(sh); return 0; }
    return sh;
}

inline GLuint makeProgram(const char* vs, const char* fs){
    GLuint v = makeShader(GL_VERTEX_SHADER,vs); if(!v) return 0;
    GLuint f = makeShader(GL_FRAGMENT_SHADER,fs); if(!f){ glDeleteShader(v); return 0; }
    GLuint p = glCreateProgram(); if(!p){ glDeleteShader(v); glDeleteShader(f); return 0; }
    glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ glDeleteProgram(p); return 0; }
    return p;
}

// VAO/VBO-Layouts einmalig konfigurieren
inline void ensurePanelVAO(GLuint& vao, GLuint& vbo){
    if(vao) return;
    glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(2*sizeof(float)));
}

inline void ensureHeatVAO(GLuint& vao, GLuint& vbo){
    if(vao) return;
    glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(float),(void*)0);
}

} // namespace UiGL
