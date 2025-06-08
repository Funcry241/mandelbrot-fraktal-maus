#pragma once

// settings.hpp — 🐭 Alle zentralen Konstanten kompakt & modern gesammelt

namespace Settings {

// 🛠️ Debugging / Test-Modus
inline constexpr bool debugGradient = false;   
inline constexpr bool debugLogging  = true;    

// 🖥️ Fenster und Bild
inline constexpr int width        = 1024;       
inline constexpr int height       = 768;        
inline constexpr int windowPosX   = 100;        
inline constexpr int windowPosY   = 100;        

// 🔎 Zoom & Pan Einstellungen
inline constexpr float initialZoom    = 3000.0f;  // 🐭 höherer Start-Zoom
inline constexpr float zoomFactor     = 1.01f;    
inline constexpr float initialOffsetX = -0.5f;    
inline constexpr float initialOffsetY =  0.0f;    

inline constexpr float OFFSET_STEP_FACTOR = 0.5f;     
inline constexpr float ZOOM_STEP_FACTOR   = 0.002f;    // 🐭 sanftere Zoomrate

inline constexpr float MIN_OFFSET_STEP = 1e-8f;       
inline constexpr float MIN_ZOOM_STEP   = 1e-6f;       

// 🧠 Auto-Zoom Steuerung
inline constexpr float VARIANCE_THRESHOLD = 1e-12f;   

// Dynamischer Variance-Threshold in Abhängigkeit vom Zoom
inline float dynamicVarianceThreshold(float zoom) {
    return VARIANCE_THRESHOLD / logf(zoom + 2.0f);
}

// 🔢 Iterations-Steuerung
inline constexpr int TILE_W             = 8;    // 🐭 feinere Kacheln
inline constexpr int TILE_H             = 8;    
inline constexpr int INITIAL_ITERATIONS = 100;  
inline constexpr int MAX_ITERATIONS_CAP = 5000; 
inline constexpr int ITERATION_STEP     = 5;    

// 🐭 Sanftes Gliding für Offset-Änderungen
inline constexpr float LERP_FACTOR      = 0.02f;  // 🐭 langsamere Zielanpassung

} // namespace Settings
