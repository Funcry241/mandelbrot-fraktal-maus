///// Otter: Public API + Back-Compat Adapters; ASCII-Logs; keine versteckte API-Drift.
///// Schneefuchs: Header/Source synchron; GL4.3-Core-Annahme; deterministisch.
///// Maus: Header schlank – keine PCH/Settings/Log-Includes; Host-Logs nur in .cpp.
///// Datei: src/renderer_resources.hpp

#pragma once
#ifndef RENDERER_RESOURCES_HPP
#define RENDERER_RESOURCES_HPP

#include <GL/glew.h> // GLuint, GLsizei

namespace OpenGLUtils {

// Kontextlabel nur für Logs in der .cpp (z. B. "init", "resize", "frame")
void   setGLResourceContext(const char* context);

// Ressourcen-Erzeugung
[[nodiscard]] GLuint createPBO(int width, int height);
[[nodiscard]] GLuint createTexture(int width, int height);

// Upload: PBO -> Texture (robust; sichert/restauriert relevanten GL-State)
void   updateTextureFromPBO(GLuint pbo, GLuint tex, int width, int height);

} // namespace OpenGLUtils

// -----------------------------------------------------------------------------
// Back-Compat: Adapter im globalen Namespace (alte Call-Sites bleiben buildbar).
// Bewahrt 0-Arg/2-Arg-Contextsetter und 3-/4-Arg-Uploadsignaturen.
// -----------------------------------------------------------------------------
void setGLResourceContext() noexcept;                                   // re-bindt zuletzt gecachte IDs
void setGLResourceContext(GLuint textureId, GLuint pboId) noexcept;     // bind + cache

void updateTextureFromPBO(GLuint textureId, GLsizei width, GLsizei height) noexcept;  // nutzt aktuell gebundenen PBO
void updateTextureFromPBO(GLuint textureId, GLuint pboId, int width, int height) noexcept; // explizit (tex, pbo, w, h)

#endif // RENDERER_RESOURCES_HPP
