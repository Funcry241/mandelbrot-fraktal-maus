//MAUS
// Lightweight framebuffer capture (single-shot, non-blocking via PBO + fence). ASCII logs only.
// Otter: async readback; Schneefuchs: state restore; logs ASCII.

#pragma once

namespace FrameCapture {
    // Call once per frame *after* the frame has been rendered (before/after swap is fine).
    // - On the 100th frame, we enqueue a non-blocking glReadPixels into a PBO.
    // - On subsequent frames, we poll the fence and write BMP once the GPU finished.
    // Cost per non-target frame: a couple of cheap branches.
    void OnFrameRendered(int frameCount);
}
