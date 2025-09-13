#pragma once
// Tiny helpers to replace ".data" usage on GL pixel buffer objects.
// NOTE: Your GLBuffer wrapper likely handles binding. If not, bind before mapping:
//   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
//   void* ptr = GLPBO::mapWrite();
//   ... write to ptr ...
//   GLPBO::unmapUnpack();

#include <GL/glew.h>

namespace GLPBO {
    inline void* mapWrite() {
        return glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    }
    inline void* mapRead() {
        return glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    }
    inline void unmapUnpack() {
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    }
    inline void unmapPack() {
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }
} // namespace GLPBO
