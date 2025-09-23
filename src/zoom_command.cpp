///// Otter: Header-only Zoom commands; host helper implemented; ASCII-only.
///// Schneefuchs: Implementiert buildAndPushZoomCommand im FramePipeline-Namespace.
///// Maus: Specht-1 – Wrapper-Ziel, entkoppelt von TU-Interna; testbar.
///// Datei: src/zoom_command.cpp

#include "pch.hpp"
#include "zoom_command.hpp"
#include "frame_context.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include <vector_types.h>
#include <vector_functions.h> // make_float2

namespace FramePipeline
{

// Wrapper-Ziel aus frame_pipeline.cpp:
//   extern void buildAndPushZoomCommand(FrameContext&, CommandBus&, int frameIndex, float zoomGain);
void buildAndPushZoomCommand(FrameContext& fctx, CommandBus& bus, int frameIndex, float zoomGain)
{
    if (!fctx.shouldZoom) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOMCMD] frame=%d no-op", frameIndex);
        }
        return;
    }

    // Vorher-Zustand sichern
    const float2 oldOffset = fctx.offset;
    const float  oldZoom   = fctx.zoom;

    // Command füllen (UI/History/Telemetry)
    ZoomCommand cmd;
    cmd.frameIndex = frameIndex;
    cmd.oldOffset  = oldOffset;
    cmd.zoomBefore = oldZoom;
    cmd.newOffset  = fctx.newOffset;
    cmd.zoomAfter  = oldZoom * zoomGain;

    // Falls ZoomCommand diese Felder hat, neutral setzen (FrameContext trägt sie nicht mehr):
    // (Kein Schaden, wenn das Struct diese Felder weiterhin definiert.)
    cmd.entropy  = 0.0f;
    cmd.contrast = 0.0f;

    bus.push(cmd);

    // FrameContext jetzt auf neuen Zustand umstellen
    fctx.offset = fctx.newOffset;
    fctx.zoom   = cmd.zoomAfter;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[ZOOMCMD] frame=%d gain=%.6f off:(%.6f,%.6f)->(%.6f,%.6f) zoom:%.6f->%.6f",
                       frameIndex, (double)zoomGain,
                       oldOffset.x, oldOffset.y,
                       fctx.newOffset.x, fctx.newOffset.y,
                       (double)oldZoom, (double)fctx.zoom);
    }
}

} // namespace FramePipeline
