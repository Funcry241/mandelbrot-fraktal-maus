// Datei: src/renderer_state.hpp
// 🦦 Otter: Einheitlich in allen Forward-Deklarationen. Keine strukturelle Überraschung.
// 🦊 Schneefuchs: Speicher & Buffer exakt definiert, feingliedrig und logisch.
// 🐜 Rote Ameise: tileSize explizit übergeben, deterministisch & sichtbar, keine impliziten Berechnungen mehr.
// 🐜 Hermelin: RAII-Wrapper für CUDA- und OpenGL-Buffer integriert.

#pragma once

#include "pch.hpp"
#include "zoom_logic.hpp" // ZoomResult für Auto-Zoom
#include <vector>
#include <string>          // 🐑 Schneefuchs: für warzenschweinText notwendig
#include "hermelin_buffer.hpp" // RAII-Wrapper

class RendererState {
public:
// 🖼️ Fensterdimensionen (OpenGL-Viewport & Framebuffer-Größe)
int width;
int height;
GLFWwindow* window = nullptr;

// 🔍 Zoomfaktor & aktueller Fraktal-Ausschnitt (in Weltkoordinaten)
double zoom = 1.0;
float2 offset = { 0.0f, 0.0f };

// 🧮 Iterationsparameter
int baseIterations = 100;  // Ausgangswert
int maxIterations  = 1000; // aktuell verwendeter Maximalwert

// 🎯 Auto-Zoom Zielkoordinaten
float2 targetOffset         = { 0.0f, 0.0f };   // analysiertes Ziel
float2 filteredTargetOffset = { 0.0f, 0.0f };   // geglättetes Ziel
float2 smoothedTargetOffset = { 0.0f, 0.0f };   // LERP-Interpoliertes Ziel
float  smoothedTargetScore  = -1.0f;            // Entropie-Score des Zieltiles (wird geglättet)

// 📈 Anzeige-Feedback
float fps = 0.0f;
float deltaTime  = 0.0f;

// 🧩 Analysepuffer (Host)
int lastTileSize = 0;
std::vector<float> h_entropy;
std::vector<float> h_contrast;

// 🔗 Analysepuffer (Device) mit RAII
Hermelin::CudaDeviceBuffer d_iterations;
Hermelin::CudaDeviceBuffer d_entropy;
Hermelin::CudaDeviceBuffer d_contrast;

// 🎥 OpenGL-Zielpuffer (Interop via CUDA) mit RAII
Hermelin::GLBuffer pbo;
Hermelin::GLBuffer tex;

// 🕒 Zeitsteuerung pro Frame
int frameCount = 0;
double lastTime = 0.0;

// 🧠 Letztes Ergebnis der Zielanalyse (persistenter Zustand)
ZoomLogic::ZoomResult zoomResult;
float lastEntropy  = 0.0f;
float lastContrast = 0.0f;

// 🔥 Heatmap-Overlay-Zustand
bool heatmapOverlayEnabled = false;

// HUD-Overlay-Zustand
bool warzenschweinOverlayEnabled = false;

// 🐑 Schneefuchs: HUD-Text für Overlay – pro Frame gesetzt, sichtbar.
std::string warzenschweinText;

// 🧽 Setup & Verwaltung
RendererState(int w, int h);
void reset();                             // stellt Initialzustand her
void setupCudaBuffers(int tileSize);      // allokiert Device-Buffer – tileSize explizit (🐜)
void resize(int newWidth, int newHeight); // Fenstergröße ändern

};
