// Datei: src/renderer_resources.hpp
///// Otter: Public API + Back-Compat Adapters; ASCII-Logs; keine versteckte API-Drift.
///// Schneefuchs: Header/Source synchron; GL4.3 Core; deterministisch.
///// Otter: Nur LUCHS_LOG_HOST im Host-Pfad; keine CUDA_CHECK-Redefinition.
#pragma once

#include "pch.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include <GL/glew.h>

namespace OpenGLUtils
{
    // Context label for logging only (e.g., "init", "resize", "frame")
    void   setGLResourceContext(const char* context);

    // Resource creation
    GLuint createPBO(int width, int height);
    GLuint createTexture(int width, int height);

    // Upload: PBO -> Texture (robust; saves/restores relevant GL state)
    void   updateTextureFromPBO(GLuint pbo, GLuint tex, int width, int height);
}

// -----------------------------------------------------------------------------
// Back-compat adapters in the global namespace to keep older call sites compiling.
// Observed call sites in frame_pipeline.cpp (today):
//   setGLResourceContext();                               // 0-arg
//   updateTextureFromPBO(GLuint, GLuint, int, int);       // 4-arg (texture, pbo, w, h)
// We also preserve the previous 2-arg context setter and 3-arg upload.
// -----------------------------------------------------------------------------
void setGLResourceContext() noexcept;                                      // no-arg (re-bind last cached IDs)
void setGLResourceContext(GLuint textureId, GLuint pboId) noexcept;        // bind + cache

void updateTextureFromPBO(GLuint textureId, GLsizei width, GLsizei height) noexcept;              // uses currently bound PBO
void updateTextureFromPBO(GLuint textureId, GLuint pboId, int width, int height) noexcept;        // explicit (texture, pbo, w, h)
