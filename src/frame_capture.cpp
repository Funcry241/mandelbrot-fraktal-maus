//MAUS
// Implementation: Async capture using GL pixel pack buffer (PBO) + fence.
// 🦦 Otter: Zero stalls on the render path — no glFinish; enqueue readback once.
// 🦊 Schneefuchs: GL state restored; ASCII-only logs; MSVC-safe fopen_s; header/source in sync.

#include "pch.hpp"              // GL headers (GLEW/GLFW etc.)
#include "frame_capture.hpp"
#include "luchs_log_host.hpp"

#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>

namespace {

struct CaptureState {
    bool     requested = false;   // set when frameCount==100
    bool     done      = false;   // finished writing file
    int      w = 0, h = 0;
    GLuint   pbo = 0;
    GLsync   fence = 0;
    size_t   bytes = 0;
} g_cap;

// --- helpers -----------------------------------------------------------------
static bool write_bmp24_from_rgba(const char* path, int w, int h, const uint8_t* rgba)
{
    if (w <= 0 || h <= 0 || !rgba) return false;

    const int rowStrideBGR = ((w * 3 + 3) / 4) * 4; // 4-byte aligned
    const uint32_t pixelDataSize = rowStrideBGR * h;
    const uint32_t fileSize = 54 + pixelDataSize;

    uint8_t hdr[54]; std::memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B'; hdr[1] = 'M';
    std::memcpy(&hdr[2],  &fileSize, 4);
    const uint32_t offBits = 54; std::memcpy(&hdr[10], &offBits, 4);
    const uint32_t dibSize = 40; std::memcpy(&hdr[14], &dibSize, 4);
    std::memcpy(&hdr[18], &w, 4);
    std::memcpy(&hdr[22], &h, 4); // positive → bottom-up
    const uint16_t planes = 1;  std::memcpy(&hdr[26], &planes, 2);
    const uint16_t bpp    = 24; std::memcpy(&hdr[28], &bpp, 2);
    std::memcpy(&hdr[34], &pixelDataSize, 4);

    // MSVC-safe file open (no C4996 under /WX)
    FILE* f = nullptr;
#if defined(_MSC_VER)
    if (fopen_s(&f, path, "wb") != 0 || !f) return false;
#else
    f = std::fopen(path, "wb");
    if (!f) return false;
#endif

    if (std::fwrite(hdr, 1, 54, f) != 54) { std::fclose(f); return false; }

    std::vector<uint8_t> row(rowStrideBGR, 0);
    // BMP is bottom-up; OpenGL rows start at y=0 bottom when height is positive in header
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgba + (size_t)y * (size_t)w * 4;
        uint8_t* dst = row.data();
        for (int x = 0; x < w; ++x) {
            const uint8_t r = src[4*x + 0];
            const uint8_t g = src[4*x + 1];
            const uint8_t b = src[4*x + 2];
            *dst++ = b; *dst++ = g; *dst++ = r; // BGR
        }
        if (std::fwrite(row.data(), 1, rowStrideBGR, f) != (size_t)rowStrideBGR) { std::fclose(f); return false; }
    }
    std::fclose(f);
    return true;
}

static void enqueue_readback()
{
    // Query viewport → width/height of the current framebuffer
    GLint vp[4] = {0,0,0,0};
    glGetIntegerv(GL_VIEWPORT, vp);
    g_cap.w = vp[2];
    g_cap.h = vp[3];
    g_cap.bytes = (size_t)g_cap.w * (size_t)g_cap.h * 4u;

    if (g_cap.w <= 0 || g_cap.h <= 0) {
        LUCHS_LOG_HOST("[CAPTURE] ERROR: invalid viewport (%d x %d)", g_cap.w, g_cap.h);
        g_cap.requested = false; g_cap.done = true;
        return;
    }

    // Create PBO if needed
    if (g_cap.pbo == 0) {
        glGenBuffers(1, &g_cap.pbo);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, g_cap.pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)g_cap.bytes, nullptr, GL_STREAM_READ);

    // Enqueue non-blocking readback from backbuffer into PBO
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, g_cap.w, g_cap.h, GL_RGBA, GL_UNSIGNED_BYTE, (void*)0);

    // Insert fence so we can poll completion later
    if (g_cap.fence) {
        glDeleteSync(g_cap.fence);
        g_cap.fence = 0;
    }
    g_cap.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    if (g_cap.fence) {
        LUCHS_LOG_HOST("[CAPTURE] Enqueued readback into PBO (w=%d h=%d bytes=%zu)", g_cap.w, g_cap.h, g_cap.bytes);
    } else {
        LUCHS_LOG_HOST("[CAPTURE] ERROR: glFenceSync failed; capture aborted");
        g_cap.requested = false; g_cap.done = true;
    }
}

static void try_finish_write()
{
    if (!g_cap.fence || g_cap.done) return;

    // Non-blocking poll
    const GLenum res = glClientWaitSync(g_cap.fence, 0, 0);
    if (res == GL_ALREADY_SIGNALED || res == GL_CONDITION_SATISFIED) {
        // Map PBO and write BMP
        glBindBuffer(GL_PIXEL_PACK_BUFFER, g_cap.pbo);
        void* ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)g_cap.bytes, GL_MAP_READ_BIT);
        if (!ptr) {
            // As a fallback, flush and wait briefly, try again
            const GLenum res2 = glClientWaitSync(g_cap.fence, GL_SYNC_FLUSH_COMMANDS_BIT, 2000000 /*2ms*/);
            (void)res2;
            ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)g_cap.bytes, GL_MAP_READ_BIT);
        }

        if (ptr) {
            const bool ok = write_bmp24_from_rgba("dist/frame_0100.bmp", g_cap.w, g_cap.h, static_cast<const uint8_t*>(ptr));
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            if (ok) {
                LUCHS_LOG_HOST("[CAPTURE] Saved 100th frame to dist/frame_0100.bmp");
            } else {
                LUCHS_LOG_HOST("[CAPTURE] ERROR: failed to write dist/frame_0100.bmp");
            }
        } else {
            LUCHS_LOG_HOST("[CAPTURE] ERROR: glMapBufferRange failed");
        }

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        // Cleanup fence and PBO (single-shot)
        glDeleteSync(g_cap.fence); g_cap.fence = 0;
        if (g_cap.pbo) { glDeleteBuffers(1, &g_cap.pbo); g_cap.pbo = 0; }
        g_cap.done = true;
    }
    // else: not ready yet → try again next frame (no stall)
}

} // anon namespace

namespace FrameCapture {

void OnFrameRendered(int frameCount)
{
    if (g_cap.done) return;

    // Trigger exactly once on the 100th rendered frame.
    if (!g_cap.requested && frameCount == 100) {
        g_cap.requested = true;
        enqueue_readback();
        return;
    }

    // If requested earlier, attempt to finish without blocking.
    if (g_cap.requested && !g_cap.done) {
        try_finish_write();
    }
}

} // namespace FrameCapture
