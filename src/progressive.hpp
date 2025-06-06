#pragma once

// progressive.hpp – Declaration for progressive Mandelbrot rendering

// Nur Deklaration (extern) — *keine* __device__ oder __managed__ hier!
extern __device__ __managed__ int currentMaxIter;
extern __device__ __managed__ bool justResetFlag;  // 🐭 Reset-Flag

void resetIterations();
int getCurrentIterations();
bool wasJustReset();  // 🐭

inline constexpr int iterStep = 50;   // Schrittweite pro Frame
inline constexpr int iterMax  = 5000; // Maximale Iterationen
