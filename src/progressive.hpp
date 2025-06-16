// Datei: src/progressive.hpp
// Zeilen: 50
// 🐭 Maus-Kommentar: Steuerungszentrale für schrittweise Iterationserhöhung im Mandelbrot-Renderer. Nutzt CUDA __managed__ Speicher für Host–Device-Synchronisation. Schneefuchs wollte es ursprünglich per Host-Callback – ich hab’s effizienter gemacht.

#pragma once
#ifndef PROGRESSIVE_HPP
#define PROGRESSIVE_HPP

// 🐭 progressive.hpp – Kontrolliert die schrittweise Erhöhung der Mandelbrot-Iterationstiefe
// ⚙️ Verwendet __managed__-Speicher für synchronisierten Zugriff zwischen Host & Device

// -----------------------------------------------------------------------------
// CUDA-Managed globale Zustandsvariablen
// -----------------------------------------------------------------------------
#ifdef __CUDACC__  // Nur verfügbar, wenn CUDA-Code kompiliert wird

// 📌 Aktuelle maximale Iterationen für Mandelbrot-Berechnung (wird progressiv erhöht)
extern __device__ __managed__ int currentMaxIter;

// 🔄 Flag für „gerade zurückgesetzt“ (nur für einen Frame gültig)
extern __device__ __managed__ bool justResetFlag;

#endif // __CUDACC__

// -----------------------------------------------------------------------------
// CPU-seitige Schnittstelle zur Steuerung (wird vom Hauptprogramm verwendet)
// -----------------------------------------------------------------------------
namespace Progressive {

    /// 🔁 Setzt Iterationen zurück auf Startwert und markiert Reset-Flag.
    void resetIterations();

    /// 🔍 Liefert aktuelle maximale Iterationstiefe (ohne Veränderung).
    int getCurrentIterations();

    /// ⏫ Erhöht Iterationstiefe schrittweise bis zur Maximalgrenze.
    void incrementIterations();

    /// 🕵️‍♂️ Gibt einmalig true zurück, wenn gerade ein Reset erfolgte (setzt Flag zurück).
    bool wasJustReset();

} // namespace Progressive

#endif // PROGRESSIVE_HPP
