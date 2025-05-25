// dd_real.cu
#include "dd_real.cuh"

__device__ inline dd_real quick_two_sum(double a, double b) {
    double s = a + b;
    double e = b - (s - a);
    return dd_real(s, e);
}

__device__ dd_real dd_add(dd_real a, dd_real b) {
    double s = a.hi + b.hi;
    double v = s - a.hi;
    double e = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    return quick_two_sum(s, e);
}

__device__ dd_real dd_sub(dd_real a, dd_real b) {
    double s = a.hi - b.hi;
    double v = s - a.hi;
    double e = (a.hi - (s - v)) - (b.hi + v) + a.lo - b.lo;
    return quick_two_sum(s, e);
}

__device__ dd_real dd_mul(dd_real a, dd_real b) {
    double p = a.hi * b.hi;
    double a1 = __hiloint2double(__double2hiint(a.hi), 0);
    double a2 = a.hi - a1;
    double b1 = __hiloint2double(__double2hiint(b.hi), 0);
    double b2 = b.hi - b1;
    double e = ((a1 * b1 - p) + a1 * b2 + a2 * b1 + a2 * b2)
             + (a.hi * b.lo + a.lo * b.hi);
    return quick_two_sum(p, e);
}

__device__ dd_real dd_div(dd_real a, dd_real b) {
    double q1 = (fabs(b.hi) < 1e-300 ? 0.0 : (a.hi / b.hi));
    dd_real q1_dd(q1);
    dd_real prod = dd_mul(b, q1_dd);
    dd_real r = dd_sub(a, prod);
    double q2 = (fabs(b.hi) < 1e-300 ? 0.0 : ((r.hi + r.lo) / b.hi));
    return dd_add(q1_dd, dd_real(q2));
}

__device__ dd_real dd_sqr(dd_real a) {
    return dd_mul(a, a);
}

__device__ double dd_abs2(dd_real x, dd_real y) {
    dd_real x2 = dd_mul(x, x);
    dd_real y2 = dd_mul(y, y);
    return x2.hi + y2.hi;
}
