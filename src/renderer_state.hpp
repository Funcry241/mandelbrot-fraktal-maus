// Datei: src/renderer_state.hpp
// Zeilen: 71
// 🐭 Maus-Kommentar: State-of-the-Art für Renderer-Status. Alle Entropie-/Kontrast-/Zoomdaten persistent und schnell (float2 statt double2). Kein toter Code: lastIndex entfernt, Übersicht und Performance jetzt maximal klar. Schneefuchs: Übersicht, Otter: Performance.
#pragma once

#include "pch.hpp" // <cuda_runtime.h>, float2 etc.
#include "zoom_logic.hpp" // ZoomResult für Auto-Zoom
#include <vector>

class RendererState {
public:
// 🖼️ Fensterdimensionen
int width;
int height;
GLFWwindow* window = nullptr;

// 🔍 Zoom & Bildausschnitt
double zoom = 1.0;
float2 offset = { 0.0f, 0.0f };

// 🧮 Iterationen
int baseIterations = 100;
int maxIterations = 1000;

// 🎯 Zielkoordinaten
double2 targetOffset = { 0.0, 0.0 };
double2 filteredTargetOffset = { 0.0, 0.0 };
float2 smoothedTargetOffset = { 0.0f, 0.0f };
float smoothedTargetScore = -1.0f;

// 📈 Anzeige
float currentFPS = 0.0f;
float deltaTime = 0.0f;

// 🧩 Entropie & Kontrast
int lastTileSize = 0;
std::vector<float> h_entropy;
std::vector<float> h_contrast;

// 🔗 CUDA-Puffer (Device)
int* d_iterations = nullptr;
float* d_entropy = nullptr;
float* d_contrast = nullptr;
int* d_tileSupersampling = nullptr;

// 🎛️ Supersampling-Puffer (Host)
std::vector<int> h_tileSupersampling;

// 🎥 OpenGL-Puffer
unsigned int pbo = 0;
unsigned int tex = 0;

// 🕒 Zeitsteuerung
int frameCount = 0;
double lastTime = 0.0;

// 🔁 Auto-Zoom
bool shouldZoom = false;

// 🧠 Analyse & Ziel
ZoomLogic::ZoomResult zoomResult;
float lastEntropy = 0.0f;
float lastContrast = 0.0f;
bool justZoomed = false;

// 📏 Globales Supersampling
int supersampling = 1;

// 🔥 Overlay
bool overlayEnabled = false;
int lastTileIndex = -1;

// 🧽 Verwaltung
RendererState(int w, int h);
void reset();
void setupCudaBuffers();
void resize(int newWidth, int newHeight);

};
