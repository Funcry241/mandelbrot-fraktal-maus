// dd_real.cuh
#pragma once
#include <cuda_runtime.h>

// Struktur für Double-Double Gleitkommazahlen
struct dd_real {
    double hi;
    double lo;

    __host__ __device__ dd_real() : hi(0.0), lo(0.0) {}
    __host__ __device__ dd_real(double h) : hi(h), lo(0.0) {}
    __host__ __device__ dd_real(double h, double l) : hi(h), lo(l) {}
};

// Nur __device__, da __double2hiint / __hiloint2double GPU-only sind
__device__ dd_real dd_add(dd_real a, dd_real b);
__device__ dd_real dd_sub(dd_real a, dd_real b);
__device__ dd_real dd_mul(dd_real a, dd_real b);
__device__ dd_real dd_div(dd_real a, dd_real b);
__device__ dd_real dd_sqr(dd_real a);
__device__ double  dd_abs2(dd_real x, dd_real y);
