///// Otter: Async Framebuffer-Capture – single-shot via PBO+Fence; zero stalls, kein glFinish.
///// Schneefuchs: Zustands-Restore & ASCII-Logs; Header/Source synchron; MSVC-safe I/O in .cpp.
///// Maus: Nach dem Render einmal pro Frame aufrufen; 100. Frame triggert Capture; Rest non-blocking.

#pragma once

namespace FrameCapture {

// Call once per frame *after* the frame has been rendered (before/after swap is fine).
// - On the 100th frame, we enqueue a non-blocking glReadPixels into a PBO.
// - On subsequent frames, we poll the fence and write BMP once the GPU finished.
// Cost per non-target frame: a couple of cheap branches.
void OnFrameRendered(int frameCount);

} // namespace FrameCapture
