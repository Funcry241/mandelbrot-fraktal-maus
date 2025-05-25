// dd_real.cuh – CUDA-kompatible Double-Double-Arithmetik
#pragma once

struct dd_real {
    double hi;
    double lo;

    __host__ __device__ dd_real() : hi(0.0), lo(0.0) {}
    __host__ __device__ dd_real(double h, double l = 0.0) : hi(h), lo(l) {}

    __host__ __device__ dd_real operator+(const dd_real& b) const {
        double s = hi + b.hi;
        double v = s - hi;
        double e = ((b.hi - v) + (hi - (s - v))) + lo + b.lo;
        return dd_real(s + e, e - (s + e - s));
    }

    __host__ __device__ dd_real operator-(const dd_real& b) const {
        return *this + dd_real(-b.hi, -b.lo);
    }

    __host__ __device__ dd_real operator*(const dd_real& b) const {
        double p = hi * b.hi;
        double e = fma(hi, b.hi, -p) + hi * b.lo + lo * b.hi;
        return dd_real(p + e, e - (p + e - p));
    }

    __host__ __device__ dd_real operator*(double b) const {
        double p = hi * b;
        double e = fma(hi, b, -p) + lo * b;
        return dd_real(p + e, e - (p + e - p));
    }

    __host__ __device__ dd_real operator/(double b) const {
        double q1 = hi / b;
        double r = ((hi - q1 * b) + lo) / b;
        return dd_real(q1 + r, r - (q1 + r - q1));
    }

    __host__ __device__ double norm2() const {
        return hi * hi + 2.0 * hi * lo;
    }

    __host__ __device__ double value() const {
        return hi;
    }
};
