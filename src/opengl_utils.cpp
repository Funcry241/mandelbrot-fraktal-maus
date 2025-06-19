// Datei: src/opengl_utils.cpp
// Zeilen: 90
// 🐭 Maus-Kommentar: Aufgeräumt. `drawFullscreenQuad()` ist raus – nur noch moderne Übergabe per VAO. Shaderfehler sauber gemeldet, Speicherverwaltung getrennt. Schneefuchs: „Nicht nur schöner – jetzt auch eindeutig!“

#include "pch.hpp"
#include "opengl_utils.hpp"

namespace OpenGLUtils {

// Hilfsfunktion: Shader kompilieren
#ifndef __CUDACC__
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512];
        glGetShaderInfoLog(s, 512, nullptr, buf);
        std::cerr << "Shader-Compile-Error: " << buf << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return s;
}

GLuint createProgramFromSource(const char* vertexSrc, const char* fragmentSrc) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, v);
    glAttachShader(prog, f);
    glLinkProgram(prog);
    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[512];
        glGetProgramInfoLog(prog, 512, nullptr, buf);
        std::cerr << "Program-Link-Error: " << buf << std::endl;
        std::exit(EXIT_FAILURE);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return prog;
}
#endif // __CUDACC__

void createFullscreenQuad(GLuint* outVAO, GLuint* outVBO, GLuint* outEBO) {
    float quad[] = {
        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f, -1.0f,   1.0f, 0.0f,
         1.0f,  1.0f,   1.0f, 1.0f,
        -1.0f,  1.0f,   0.0f, 1.0f
    };
    unsigned idx[] = { 0, 1, 2, 2, 3, 0 };

    glGenVertexArrays(1, outVAO);
    glGenBuffers(1, outVBO);
    glGenBuffers(1, outEBO);

    glBindVertexArray(*outVAO);
    glBindBuffer(GL_ARRAY_BUFFER, *outVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *outEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void deleteFullscreenQuad(GLuint* inVAO, GLuint* inVBO, GLuint* inEBO) {
    glDeleteBuffers(1, inVBO);
    glDeleteBuffers(1, inEBO);
    glDeleteVertexArrays(1, inVAO);
}

} // namespace OpenGLUtils
