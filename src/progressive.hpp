#pragma once

// 🐭 progressive.hpp – Deklaration für progressives Mandelbrot-Rendering

#ifdef __CUDACC__
// Für CUDA-Compiler (Device und Host gemeinsam)
__device__ __managed__ inline int currentMaxIter = 100;
#else
// Für Host-Compiler (Deklaration)
extern int currentMaxIter;
#endif

extern void resetIterations();
extern int getCurrentIterations();  // 🐭 Getter-Deklaration

// 🐭 Konstante Schrittweiten (modern, constexpr)
inline constexpr int iterStep = 50;          // Schrittweite pro Frame
inline constexpr int iterMax  = 5000;        // Maximale Iterationen
