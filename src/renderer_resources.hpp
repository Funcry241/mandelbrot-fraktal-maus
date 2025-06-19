// Datei: src/opengl_utils.hpp
// Zeilen: 22
// 🐭 Maus-Kommentar: Stellt OpenGL-Helfer bereit – Erstellung von PBOs und Texturen für das Fraktal-Rendering. Muss in `renderer_resources.cpp` sichtbar sein. Schneefuchs: „Ohne diese Helfer meckert der Linker – wie ein Otter ohne Wasser!“

#pragma once

#include "pch.hpp"

namespace OpenGLUtils {

// 🖼️ Erstellt einen Pixel Buffer Object (PBO) für die CUDA/OpenGL-Interop
GLuint createPBO(int width, int height);

// 🎨 Erstellt eine OpenGL-Textur zur Darstellung im Shader
GLuint createTexture(int width, int height);

} // namespace OpenGLUtils
