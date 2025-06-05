#pragma once

// 🐭 progressive.hpp – Deklaration für progressives Mandelbrot-Rendering

extern void resetIterations();

// 🐭 Laufzeitvariable
inline int currentMaxIter = 100;             // Startwert (wird dynamisch verändert)

// 🐭 Konstante Schrittweiten (modern, constexpr)
inline constexpr int iterStep = 50;          // Schrittweite pro Frame
inline constexpr int iterMax  = 5000;        // Maximale Iterationen
