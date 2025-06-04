// Datei: src/opengl_utils.hpp
#ifndef OPENGL_UTILS_HPP
#define OPENGL_UTILS_HPP

#ifndef __CUDACC__
#include <GL/glew.h>
#endif


// Erstellt ein Shader‐Programm aus Quelltexten (Vertex & Fragment)
GLuint createProgramFromSource(const char* vertexSrc, const char* fragmentSrc);

// Erstellt einen Vollbild‐Quad mit VAO, VBO, EBO
void createFullscreenQuad(GLuint* outVAO, GLuint* outVBO, GLuint* outEBO);

// Zeichnet das Quad (setzt VAO voraus)
void drawFullscreenQuad();

// Löscht die Ressourcen des Quad
void deleteFullscreenQuad(GLuint* inVAO, GLuint* inVBO, GLuint* inEBO);

#endif // OPENGL_UTILS_HPP
