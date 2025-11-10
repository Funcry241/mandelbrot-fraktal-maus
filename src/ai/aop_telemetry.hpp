///// Otter: AOP Telemetry — zentrale Deklarationen für HUD/Overlays
///// Schneefuchs: Externe Symbole mit klaren Typen; ASCII-only; /WX-safe
///// Maus: Nur Deklarationen; Definitionen in aop_telemetry.cpp
///// Datei: src/ai/aop_telemetry.hpp
#pragma once

namespace AOP_Telemetry
{
    // Policy-Ziel (NDC)
    extern float g_ai_ndc_pol_x;
    extern float g_ai_ndc_pol_y;

    // Overlay-Preview (NDC, z. B. skaliert)
    extern float g_ai_ndc_ovl_x;
    extern float g_ai_ndc_ovl_y;

    // Gültigkeitsflag für Overlay-Markierung
    extern int   g_ai_ov_valid;

    // Letzte Distanz (diagnostisch; -1.0f wenn nicht berechnet)
    extern float g_ai_last_delta;

    // Confidence of current AI pick [0..1]
    extern float g_ai_confidence;

    // Monoton steigende Zähler-ID für Policy-Evaluierungen (diagnostisch)
    extern unsigned long long g_ai_frame_id;
}
