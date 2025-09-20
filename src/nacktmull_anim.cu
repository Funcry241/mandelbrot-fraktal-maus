///// Otter: Definiert Animations-Uniforms (g_sinA/g_sinB) in __constant__.
///// Schneefuchs: Eigene TU → ODR-sicher; Single-Responsibility; /WX-fest.
///// Maus: Winzig, ohne Logs; keine Host-Abhängigkeiten.
///// Datei: src/nacktmull_anim.cu

#include <cuda_runtime.h>

__constant__ float g_sinA = 0.0f;  // ~sin(0.30*t)
__constant__ float g_sinB = 0.0f;  // ~sin(0.80*t)
