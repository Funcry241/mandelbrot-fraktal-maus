#pragma once

// 🐭 progressive.hpp – Declaration for progressive Mandelbrot rendering

#ifdef __CUDACC__
// Für CUDA-Compiler (Device und Host gemeinsam)
__device__ __managed__ inline int currentMaxIter = 100;
__device__ __managed__ inline bool justResetFlag = false;   // 🐭 NEU: Flag für Reset
#else
// Für Host-Compiler (Deklaration)
extern int currentMaxIter;
extern bool justResetFlag;                                  // 🐭 NEU: Flag für Reset
#endif

extern void resetIterations();
extern int getCurrentIterations();
extern bool wasJustReset();    // 🐭 NEU

// 🐭 Konstante Schrittweiten (modern, constexpr)
inline constexpr int iterStep = 50;          // Schrittweite pro Frame
inline constexpr int iterMax  = 5000;        // Maximale Iterationen
