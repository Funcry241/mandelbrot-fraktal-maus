// Datei: src/opengl_utils.cpp
// 🐭 Maus-Kommentar: Shaderfehler beenden das Programm nicht mehr, sondern geben 0 zurück und loggen klar. Schneefuchs: „Otter stirbt erst, wenn du willst.“

#include "pch.hpp"
#include "opengl_utils.hpp"

namespace OpenGLUtils {

// Hilfsfunktion: Shader kompilieren
static GLuint compileShader(GLenum type, const char* src) {
GLuint s = glCreateShader(type);
glShaderSource(s, 1, &src, nullptr);
glCompileShader(s);
GLint ok;
glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
if (!ok) {
char buf[512];
glGetShaderInfoLog(s, 512, nullptr, buf);
std::cerr << "[ShaderError] Compilation failed ("
<< (type == GL_VERTEX_SHADER ? "Vertex" : "Fragment")
<< "): " << buf << std::endl;
glDeleteShader(s);
return 0;
}
return s;
}

GLuint createProgramFromSource(const char* vertexSrc, const char* fragmentSrc) {
GLuint v = compileShader(GL_VERTEX_SHADER, vertexSrc);
if (v == 0) return 0;
GLuint f = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
if (f == 0) { glDeleteShader(v); return 0; }

GLuint prog = glCreateProgram();
glAttachShader(prog, v);
glAttachShader(prog, f);
glLinkProgram(prog);
GLint ok;
glGetProgramiv(prog, GL_LINK_STATUS, &ok);
if (!ok) {
    char buf[512];
    glGetProgramInfoLog(prog, 512, nullptr, buf);
    std::cerr << "[ShaderError] Program link failed: " << buf << std::endl;
    glDeleteShader(v);
    glDeleteShader(f);
    glDeleteProgram(prog);
    return 0;
}
glDeleteShader(v);
glDeleteShader(f);
return prog;

}

void createFullscreenQuad(GLuint* outVAO, GLuint* outVBO, GLuint* outEBO) {
constexpr float quad[] = {
-1.0f, -1.0f, 0.0f, 0.0f,
1.0f, -1.0f, 1.0f, 0.0f,
1.0f, 1.0f, 1.0f, 1.0f,
-1.0f, 1.0f, 0.0f, 1.0f
};
constexpr unsigned idx[] = { 0, 1, 2, 2, 3, 0 };

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

} // namespace OpenGLUtils
