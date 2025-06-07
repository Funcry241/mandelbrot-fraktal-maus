#pragma once

// progressive.hpp – 🐭 Declaration für progressive Mandelbrot-Iteration (keine Konstanten!)

// ----------------------------------------------------------------------
// Device-Managed globale Variablen für Iterationssteuerung
extern __device__ __managed__ int currentMaxIter;  // Aktuelle maximale Iterationen
extern __device__ __managed__ bool justResetFlag;  // Reset-Flag (true, wenn Reset passiert ist)

// ----------------------------------------------------------------------
// Funktionen zur Steuerung der Iterationen
void resetIterations();       // Setzt Iterationen auf Initialwert zurück und setzt Reset-Flag
int  getCurrentIterations();  // Gibt aktuelle Iterationszahl zurück (ohne Erhöhung)
void incrementIterations();   // Erhöht Iterationen progressiv bis zur Maximalgrenze
bool wasJustReset();          // Liefert true einmalig nach einem Reset
