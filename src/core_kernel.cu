// Datei: src/core_kernel.cu
// Maus-Kommentar: Einziger .cu-File mit switch-case Dispatcher. Redundanzen zulässig für SM-Divergenz.

#include "core_kernel.h"
#include <cstdio>
#include <cufft.h>

// Beispiel-Implementierungen (stark vereinfacht)
__device__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__device__ void bfsKernel(int* graph, int* dist, int V) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < V) {
        // Dummy BFS: initialisiere Distanz
        dist[idx] = (idx == 0) ? 0 : -1;
    }
}

__device__ void fftKernel(cufftComplex* data, int N) {
    // Platzhalter für FFT-Operation
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx].x *= 1.0f;
        data[idx].y *= 1.0f;
    }
}

__device__ void customKernel(void* in, void* out, int sz) {
    // Benutzerdefinierte Berechnung als Beispiel
    char* input  = reinterpret_cast<char*>(in);
    char* output = reinterpret_cast<char*>(out);
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < sz) {
        output[idx] = input[idx] ^ 0xFF;
    }
}

__global__ void unifiedKernel(Task* tasks, int numTasks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTasks) return;

    Task t = tasks[tid];
    switch (t.id) {
        case TASK_MATRIX_MUL: {
            int N = t.size;
            float* A = reinterpret_cast<float*>(t.input);
            float* B = reinterpret_cast<float*>(t.output); // temporär B
            float* C = reinterpret_cast<float*>(t.output);
            dim3 threads(16,16);
            dim3 blocks((N+15)/16,(N+15)/16);
            matrixMulKernel<<<blocks, threads>>>(A, B, C, N);
            break;
        }
        case TASK_BFS: {
            int V = t.size;
            int* graph = reinterpret_cast<int*>(t.input);
            int* dist  = reinterpret_cast<int*>(t.output);
            bfsKernel<<<(V+255)/256,256>>>(graph, dist, V);
            break;
        }
        case TASK_FFT: {
            int N = t.size;
            cufftComplex* data = reinterpret_cast<cufftComplex*>(t.input);
            fftKernel<<<(N+255)/256,256>>>(data, N);
            break;
        }
        case TASK_CUSTOM: {
            int sz = t.size;
            customKernel<<<(sz+255)/256,256>>>(t.input, t.output, sz);
            break;
        }
        default:
            // Unbekannte Task
            break;
    }
}
