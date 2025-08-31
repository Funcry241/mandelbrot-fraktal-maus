///// Otter: OpenGL-Utils (Header) – schlank; nur GLuint-Typ, klare API.
///// Schneefuchs: Header/Source synchron; deterministisch; ASCII-only; keine Seiteneffekte.
///// Maus: GLEW/GLFW im PCH ok; hier minimal halten – sonst beißt die Robbe!
///// Datei: src/opengl_utils.hpp

#pragma once
#ifndef OPENGL_UTILS_HPP
#define OPENGL_UTILS_HPP

#include <GL/glew.h> // Für GLuint (leichtgewichtig genug; sonst via PCH eingebunden)

namespace OpenGLUtils {

// 🎨 Shader-Utilities
[[nodiscard]] GLuint createProgramFromSource(const char* vertexSrc, const char* fragmentSrc);

// 🖼️ Fullscreen-Quad-Utilities
void createFullscreenQuad(GLuint* outVAO, GLuint* outVBO, GLuint* outEBO);

} // namespace OpenGLUtils

#endif // OPENGL_UTILS_HPP
