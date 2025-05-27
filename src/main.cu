// Datei: src/main.cpp
// Maus-Kommentar: Host-Code initialisiert Tasks, überträgt Daten zur GPU und startet den unifiedKernel; synchronisiert und bereinigt alles.

#include "core_kernel.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <vector>

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int numTasks = 4;
    std::vector<Task> tasks(numTasks);

    // Beispiel-Größe für alle Tasks
    int N = 1024;

    // Matrix-Multiplikation-Daten
    float *h_A, *h_B, *h_C;
    checkCuda(cudaMallocHost(&h_A, N*N*sizeof(float)), "HostAlloc A");
    checkCuda(cudaMallocHost(&h_B, N*N*sizeof(float)), "HostAlloc B");
    checkCuda(cudaMallocHost(&h_C, N*N*sizeof(float)), "HostAlloc C");
    // (Initialisierung von h_A, h_B, h_C hier ...)

    tasks[0].id     = TASK_MATRIX_MUL;
    tasks[0].input  = h_A;
    tasks[0].output = h_C;
    tasks[0].size   = N;

    // BFS-Daten
    int *h_graph, *h_dist;
    checkCuda(cudaMallocHost(&h_graph, N * sizeof(int)), "HostAlloc graph");
    checkCuda(cudaMallocHost(&h_dist,  N * sizeof(int)), "HostAlloc dist");
    // (Initialisierung von h_graph hier ...)

    tasks[1].id     = TASK_BFS;
    tasks[1].input  = h_graph;
    tasks[1].output = h_dist;
    tasks[1].size   = N;

    // FFT-Daten
    cufftComplex *h_data;
    checkCuda(cudaMallocHost(&h_data, N * sizeof(cufftComplex)), "HostAlloc data");
    // (Initialisierung von h_data hier ...)

    tasks[2].id     = TASK_FFT;
    tasks[2].input  = h_data;
    tasks[2].output = nullptr;
    tasks[2].size   = N;

    // Custom-Task-Daten
    char *h_in, *h_out;
    checkCuda(cudaMallocHost(&h_in, N), "HostAlloc in");
    checkCuda(cudaMallocHost(&h_out, N), "HostAlloc out");
    // (Initialisierung von h_in hier ...)

    tasks[3].id     = TASK_CUSTOM;
    tasks[3].input  = h_in;
    tasks[3].output = h_out;
    tasks[3].size   = N;

    // GPU-Speicher für Task-Array anlegen und kopieren
    Task* d_tasks;
    checkCuda(cudaMalloc(&d_tasks, numTasks * sizeof(Task)), "CudaMalloc tasks");
    checkCuda(cudaMemcpy(d_tasks, tasks.data(),
                        numTasks * sizeof(Task),
                        cudaMemcpyHostToDevice),
              "Memcpy tasks");

    // Kernel-Aufruf
    unifiedKernel<<<(numTasks+255)/256, 256>>>(d_tasks, numTasks);
    checkCuda(cudaDeviceSynchronize(), "Kernel launch");

    std::cout << "Alle Tasks abgeschlossen." << std::endl;

    // Cleanup
    cudaFree(d_tasks);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
    cudaFreeHost(h_graph); cudaFreeHost(h_dist);
    cudaFreeHost(h_data);
    cudaFreeHost(h_in); cudaFreeHost(h_out);

    return 0;
}
