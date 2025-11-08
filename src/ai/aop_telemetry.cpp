///// Otter: AOP Telemetry — Definitionen der externen Variablen
///// Schneefuchs: Einzige TU mit ODR-Definition; ASCII-only; /WX-safe
///// Maus: Defaults auf 0 bzw. -1; keine Abhängigkeiten
///// Datei: src/ai/aop_telemetry.cpp
#include "ai/aop_telemetry.hpp"

namespace AOP_Telemetry
{
    float g_ai_ndc_pol_x = 0.0f;
    float g_ai_ndc_pol_y = 0.0f;

    float g_ai_ndc_ovl_x = 0.0f;
    float g_ai_ndc_ovl_y = 0.0f;

    int   g_ai_ov_valid  = 0;

    float g_ai_last_delta = -1.0f;
}
