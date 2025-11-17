///// Otter: Mandelbrot-Iteration als ASM-Basis; skalarer Punkt-Evaluator für CPU-Pfade.
///// Schneefuchs: Klare C-ABI-Signatur; keine Includes, keine STL; Header-only Deklaration.
///// Maus: Einstiegspunkt für spätere Zeilen-/Tile-Renderer und Debug-Vergleich mit C++-Referenz.
///// Datei: src/asm/mandelbrot_iter.hpp
#pragma once

// Scalar Mandelbrot iteration implemented in src/asm/mandelbrot_iter.asm.
//
// Contract:
//   - x0, y0: coordinates of c in the complex plane (double).
//   - maxIters: positive iteration cap.
//   - return: number of iterations performed until escape or maxIters if not escaped.
//   - escape radius: |z|^2 > 4.0.
extern "C" int mandelbrotIter_asm(double x0, double y0, int maxIters);
