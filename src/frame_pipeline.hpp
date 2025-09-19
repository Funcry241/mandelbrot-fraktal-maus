///// Otter: Pipeline-API schlank – ein öffentlicher Einstiegspunkt: execute(RendererState&).
///// Schneefuchs: Header/Source synchron; keine schweren Includes; keine verdeckten Abhängigkeiten.
///// Maus: Zoom V2 bleibt intern in der .cpp (deterministische Reihenfolge, einheitliche Tiles/Upload).
///// Datei: src/frame_pipeline.hpp

#pragma once
#ifndef FRAME_PIPELINE_HPP
#define FRAME_PIPELINE_HPP

// Bewusst nur Vorwärtsdeklaration – Header leichtgewichtig halten.
class RendererState;

namespace FramePipeline {

// Komplettes Frame ausführen (begin -> CUDA/Analyse -> Zoom-Step -> Draw -> Perf-Logs).
// Exponiert nur den stabilen Einstiegspunkt; alle Teilphasen bleiben intern
// (siehe src/frame_pipeline.cpp), um Header/Source driftfrei zu halten.
void execute(RendererState& state);

} // namespace FramePipeline

#endif // FRAME_PIPELINE_HPP
