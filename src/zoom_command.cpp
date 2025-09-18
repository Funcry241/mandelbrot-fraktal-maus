///// Otter: Header-only Zoom commands; host helper implemented; ASCII-only.
/// ///// Schneefuchs: Implementiert buildAndPushZoomCommand im FramePipeline-Namespace.
/// ///// Maus: Specht-1 – Wrapper-Ziel, entkoppelt von TU-Interna; testbar.
/// ///// Datei: src/zoom_command.cpp

#include "pch.hpp"
#include "zoom_command.hpp"
#include "frame_context.hpp"

#include <vector_types.h>
#include <vector_functions.h> // make_float2

namespace FramePipeline
{

// Wrapper-Ziel aus frame_pipeline.cpp:
//   extern void buildAndPushZoomCommand(FrameContext&, CommandBus&, int frameIndex, float zoomGain);
void buildAndPushZoomCommand(FrameContext& fctx, CommandBus& bus, int frameIndex, float zoomGain)
{
    if (!fctx.shouldZoom) return;

    const double2 diff = {
        fctx.newOffset.x - fctx.offset.x,
        fctx.newOffset.y - fctx.offset.y
    };
    const float prevZoom = fctx.zoom;

    fctx.offset = fctx.newOffset;
    fctx.zoom  *= zoomGain;

    ZoomCommand cmd;
    cmd.frameIndex = frameIndex;
    cmd.oldOffset  = make_float2(
        static_cast<float>(fctx.offset.x - diff.x),
        static_cast<float>(fctx.offset.y - diff.y)
    );
    cmd.zoomBefore = prevZoom;
    cmd.newOffset  = make_float2(
        static_cast<float>(fctx.newOffset.x),
        static_cast<float>(fctx.newOffset.y)
    );
    cmd.zoomAfter  = fctx.zoom;
    cmd.entropy    = fctx.lastEntropy;
    cmd.contrast   = fctx.lastContrast;

    bus.push(cmd);
    fctx.timeSinceLastZoom = 0.0f;
}

} // namespace FramePipeline
