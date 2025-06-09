#pragma once
#ifndef PROGRESSIVE_HPP
#define PROGRESSIVE_HPP

// 🐭 progressive.hpp – Steuerung der progressiven Mandelbrot-Iterationen (CUDA Managed Memory)

// ----------------------------------------------------------------------
// Device-Managed globale Variablen (nur CUDA, Forward-Deklaration)
#ifdef __CUDACC__
extern __device__ __managed__ int currentMaxIter;   // Aktuelle Iterationsgrenze
extern __device__ __managed__ bool justResetFlag;   // Reset-Flag
#endif

// ----------------------------------------------------------------------
// Progressive Iteration Control (thread-safe via managed memory)

namespace Progressive {

/// Setzt Iterationen auf Startwert zurück und aktiviert Reset-Flag.
void resetIterations();

/// Liefert aktuelle Iterationszahl (ohne Erhöhung).
int getCurrentIterations();

/// Erhöht Iterationszahl schrittweise (bis Maximalgrenze).
void incrementIterations();

/// Gibt true zurück, wenn zuletzt ein Reset stattfand (einmalig pro Reset).
bool wasJustReset();

} // namespace Progressive

#endif // PROGRESSIVE_HPP
