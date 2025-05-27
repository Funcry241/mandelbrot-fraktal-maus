// Datei: src/core_kernel.h
// Maus-Kommentar: Header legt Task-Strukturen und Dispatcher fest; mappt Instruktions-Sets auf einen Kern für maximale Abdeckung.

#ifndef CORE_KERNEL_H
#define CORE_KERNEL_H

#include <cuda_runtime.h>

// Task-IDs für verschiedene Berechnungen
enum TaskID {
    TASK_MATRIX_MUL = 0,
    TASK_BFS         = 1,
    TASK_FFT         = 2,
    TASK_CUSTOM      = 3
};

// Struktur zur Übergabe von Task-Daten an den Kernel
struct Task {
    int id;         // Identifiziert die Rechenaufgabe
    void* input;    // Zeiger auf Eingabedaten
    void* output;   // Zeiger auf Ausgabedaten
    int   size;     // Größe bzw. Dimension der Aufgabe
};

// Dispatcher-Kernel: ruft je nach TaskID die jeweilige Device-Funktion auf
__global__ void unifiedKernel(Task* tasks, int numTasks);

#endif // CORE_KERNEL_H
