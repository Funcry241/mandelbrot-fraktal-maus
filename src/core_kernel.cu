// Datei: src/core_kernel.cu
// Maus-Kommentar: Einziger .cu-File mit switch-case Dispatcher und dynamischer Parallelität via nested kernel launches.

#include "core_kernel.h"
#include <cstdio>
#include <cufft.h>

// Beispiel-Implementierungen (stark vereinfacht) als __global__ Kernels für nested launches
__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
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

__global__ void bfsKernel(int* graph, int* dist, int V) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < V) {
        dist[idx] = (idx == 0) ? 0 : -1;
    }
}

__global__ void fftKernel(cufftComplex* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx].x *= 1.0f;
        data[idx].y *= 1.0f;
    }
}

__global__ void customKernel(void* in, void* out, int sz) {
    char* input  = reinterpret_cast<char*>(in);
    char* output = reinterpret_cast<char*>(out);
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < sz) {
        output[idx] = input[idx] ^ 0xFF;
    }
}

// Dispatcher-Kernel: ruft nested launches auf
__global__ void unifiedKernel(Task* tasks, int numTasks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTasks) return;

    Task t = tasks[tid];
    switch (t.id) {
        case TASK_MATRIX_MUL: {
            int N = t.size;
            dim3 threads(16,16);
            dim3 blocks((N+threads.x-1)/threads.x, (N+threads.y-1)/threads.y);
            matrixMulKernel<<<blocks, threads>>>(
                reinterpret_cast<float*>(t.input),
                reinterpret_cast<float*>(t.input),
                reinterpret_cast<float*>(t.output),
                N
            );
            break;
        }
        case TASK_BFS: {
            int V = t.size;
            int blocks = (V + 255) / 256;
            bfsKernel<<<blocks, 256>>>(
                reinterpret_cast<int*>(t.input),
                reinterpret_cast<int*>(t.output),
                V
            );
            break;
        }
        case TASK_FFT: {
            int N = t.size;
            int blocks = (N + 255) / 256;
            fftKernel<<<blocks, 256>>>(
                reinterpret_cast<cufftComplex*>(t.input),
                N
            );
            break;
        }
        case TASK_CUSTOM: {
            int sz = t.size;
            int blocks = (sz + 255) / 256;
            customKernel<<<blocks, 256>>>(t.input, t.output, sz);
            break;
        }
        default:
            // Unbekannte Task
            break;
    }
}
