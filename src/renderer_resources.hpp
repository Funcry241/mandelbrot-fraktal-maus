// Datei: src/renderer_resources.hpp
// Zeilen: 28
// ⏱️ Modular und mit Logging-Kontext – Ressourcenursprung immer nachvollziehbar.

#pragma once
#ifndef RENDERER_RESOURCES_HPP
#define RENDERER_RESOURCES_HPP

#ifdef CUDACC
typedef unsigned int GLuint;
#else
#include <GL/glew.h>
#endif

namespace OpenGLUtils {

// 🔧 Setzt den Kontext-String für folgende Ressourcen-Log-Ausgaben (z. B. "resize", "init").
void setGLResourceContext(const char* context);

// 🧱 Erzeugt OpenGL-Buffer/Texture (mit Logging bei Bedarf)
GLuint createPBO(int width, int height);
GLuint createTexture(int width, int height);

} // namespace OpenGLUtils

#endif // RENDERER_RESOURCES_HPPs
