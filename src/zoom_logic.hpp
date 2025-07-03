// Datei: src/zoom_logic.hpp  
// Zeilen: 50  
// 🐭 Maus-Kommentar: Nur noch Deklarationen! Für saubere Trennung von Interface und Implementation.  
// Flugente-konform: Koordinaten sind wieder float2 – schnell, CUDA-freundlich.  
// Schneefuchs: „Header macht Angebot, nicht Geschäft.“  

#pragma once  
#include "common.hpp"  
#include "settings.hpp"  
#include <vector>  
#include <vector_types.h> // für float2  

namespace ZoomLogic {  

struct ZoomResult {  
    int bestIndex = -1;  
    float bestEntropy = 0.0f;  
    float bestContrast = 0.0f;  
    float bestScore = 0.0f;  
    float distance = 0.0f;  
    float minDistance = 0.0f;  
    float relEntropyGain = 0.0f;  
    float relContrastGain = 0.0f;  
    bool isNewTarget = false;  
    bool shouldZoom = false;  
    float2 newOffset = make_float2(0.0f, 0.0f);  // Flugente!  

    std::vector<float> perTileContrast;  // 🔥 Kontrastwerte für HeatmapOverlay  
};  

// 🧠 Berechnung: mittlerer Kontrast aus Nachbarentropien  
float computeEntropyContrast(const std::vector<float>& entropy, int width, int height, int tileSize);  

// 🧠 Entscheidung: neues Ziel auswählen (Flugente-Version, 13 Argumente)  
ZoomResult evaluateZoomTarget(  
    const std::vector<float>& entropy,  
    const std::vector<float>& contrast,  
    float2 currentOffset,   // Flugente!  
    float zoom,  
    int width,  
    int height,  
    int tileSize,  
    float2 previousOffset,  // Flugente!  
    int previousIndex,  
    float previousEntropy,  
    float previousContrast  
);  

} // namespace ZoomLogic
