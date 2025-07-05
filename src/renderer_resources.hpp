// Datei: src/renderer_resources.hpp
// Zeilen: 28
// ⏱️ Nach wie vor modularisiert – jetzt mit Kontextsteuerung für Logging

#pragma once
#ifndef RENDERER_RESOURCES_HPP
#define RENDERER_RESOURCES_HPP

#ifdef __CUDACC__
typedef unsigned int GLuint;
#else
#include <GL/glew.h>
#endif

namespace OpenGLUtils {

// 🔧 Kontext für Logging – z. B. "resize", "init", "tileSizeChange"
void setGLResourceContext(const char* context);

// 🧱 OpenGL-Ressourcen erzeugen
GLuint createPBO(int width, int height);
GLuint createTexture(int width, int height);

} // namespace OpenGLUtils

#endif // RENDERER_RESOURCES_HPP
