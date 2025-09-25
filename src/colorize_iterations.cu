///// Otter: Iteration→PBO colorizer + GPU heatmap metrics; compact, branch-light.
///// Schneefuchs: Uses Settings block geometry; stable HSV ramp with gentle gamma.
///// Maus: One colorize kernel + one metrics kernel; clamp + pack to RGBA; minimal deps.
///// Datei: src/colorize_iterations.cu

#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include <vector>   // for metrics host export

#include "settings.hpp"
#include "colorize_iterations.cuh"

// ----------------------------- tiny math helpers ------------------------------
static __device__ __forceinline__ float clamp01(float x) {
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

static __device__ __forceinline__ float3 hsv_to_rgb(float h, float s, float v) {
    // h in [0,1), s,v in [0,1]
    float r, g, b;
    float i = floorf(h * 6.0f);
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    int   ii = static_cast<int>(i) % 6;
    if      (ii == 0) { r=v; g=t; b=p; }
    else if (ii == 1) { r=q; g=v; b=p; }
    else if (ii == 2) { r=p; g=v; b=t; }
    else if (ii == 3) { r=p; g=q; b=v; }
    else if (ii == 4) { r=t; g=p; b=v; }
    else              { r=v; g=p; b=q; }
    return make_float3(r, g, b);
}

static __device__ __forceinline__ uchar4 pack_rgba(float r, float g, float b, float a=1.0f) {
    r = clamp01(r); g = clamp01(g); b = clamp01(b); a = clamp01(a);
    return make_uchar4(static_cast<unsigned char>(r * 255.0f + 0.5f),
                       static_cast<unsigned char>(g * 255.0f + 0.5f),
                       static_cast<unsigned char>(b * 255.0f + 0.5f),
                       static_cast<unsigned char>(a * 255.0f + 0.5f));
}

// ------------------------------ palette mapping ------------------------------
// A smooth HSV ramp based on normalized iteration count with light easing.
// Interior pixels (it == maxIter) are rendered as near-black for "Rüsselwarze" look.
static __device__ __forceinline__ uchar4 color_from_iter(uint16_t it, int maxIter) {
    if (it >= static_cast<uint16_t>(maxIter)) {
        // Interior: keep it very dark but not pure black to preserve subtle gradients if needed
        const float v = 0.02f;
        return pack_rgba(v, v, v, 1.0f);
    }

    // Normalize and ease a bit to stretch low iterations
    const float t0 = static_cast<float>(it) / static_cast<float>(maxIter);
    const float t  = powf(t0, 0.85f); // gentle gamma

    // Hue cycles softly over a limited band to avoid rainbow noise
    const float hue = fmodf(0.62f + 0.38f * t * 1.25f, 1.0f); // 0.62..~1.0
    const float sat = 0.78f;
    const float val = 0.98f * (0.35f + 0.65f * t);            // lift with t

    const float3 rgb = hsv_to_rgb(hue, sat, val);
    return pack_rgba(rgb.x, rgb.y, rgb.z, 1.0f);
}

// ---------------------------------- kernel -----------------------------------
__global__ void kColorizeIterationsToPBO(
    const uint16_t* __restrict__ d_it,
    uchar4*       __restrict__   d_out,
    int                          width,
    int                          height,
    int                          maxIter
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int idx = y * width + x;

    const uint16_t it = d_it[idx];
    d_out[idx] = color_from_iter(it, maxIter);
}

// ---------------------------------- launch -----------------------------------
extern "C" void colorize_iterations_to_pbo(
    const uint16_t* d_iterations,
    uchar4*         d_pboOut,
    int             width,
    int             height,
    int             maxIter,
    cudaStream_t    stream
) noexcept
{
    if (!d_iterations || !d_pboOut || width <= 0 || height <= 0 || maxIter <= 0) {
        return;
    }

    dim3 block(Settings::MANDEL_BLOCK_X, Settings::MANDEL_BLOCK_Y, 1);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              1);

    kColorizeIterationsToPBO<<<grid, block, 0, stream>>>(
        d_iterations, d_pboOut, width, height, maxIter
    );
}

// ============================================================================
// GPU Heatmap Metrics (entropy/contrast) – one tile per block, 1 thread/block
// ============================================================================

__global__ void colorize_kernel_tile_metrics(const uint16_t* __restrict__ it,
                                             int w, int h,
                                             int tilePx, int tilesX,
                                             float* __restrict__ entropy,
                                             float* __restrict__ contrast)
{
    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    const int tilesY = gridDim.y;
    if (tx >= tilesX || ty >= tilesY) return;

    const int x0 = tx * tilePx;
    const int y0 = ty * tilePx;
    const int x1 = (x0 + tilePx > w) ? w : (x0 + tilePx);
    const int y1 = (y0 + tilePx > h) ? h : (y0 + tilePx);

    const int tileW = (x1 - x0 > 0) ? (x1 - x0) : 0;
    const int tileH = (y1 - y0 > 0) ? (y1 - y0) : 0;
    const int nPix  = tileW * tileH;
    const int outIx = ty * tilesX + tx;

    if (nPix <= 0) {
        if (entropy)  entropy[outIx]  = 0.0f;
        if (contrast) contrast[outIx] = 0.0f;
        return;
    }

    // Contrast = stddev over the tile
    double sum = 0.0, sum2 = 0.0;
    for (int y = y0; y < y1; ++y) {
        const uint16_t* row = it + (size_t)y * (size_t)w + x0;
        for (int x = 0; x < tileW; ++x) {
            const double v = (double)row[x];
            sum  += v;
            sum2 += v * v;
        }
    }
    const double mean = sum / (double)nPix;
    double var = sum2 / (double)nPix - mean * mean;
    if (var < 0.0) var = 0.0;
    const float stdev = (float)sqrt(var);
    if (contrast) contrast[outIx] = stdev;

    // Entropy via 32 fixed buckets (hash-based, iterations-agnostic)
    constexpr int B = 32;
    int hist[B];
    #pragma unroll
    for (int i = 0; i < B; ++i) hist[i] = 0;

    for (int y = y0; y < y1; ++y) {
        const uint16_t* row = it + (size_t)y * (size_t)w + x0;
        for (int x = 0; x < tileW; ++x) {
            const uint16_t v = row[x];
            const int b = ((int)v ^ ((int)v >> 5)) & (B - 1);
            ++hist[b];
        }
    }

    float H = 0.0f;
    const float invN  = 1.0f / (float)nPix;
    constexpr float invLn2 = 1.0f / 0.6931471805599453f;
    for (int i = 0; i < B; ++i) {
        const float p = (float)hist[i] * invN;
        if (p > 0.0f) H -= p * (logf(p) * invLn2);
    }
    if (entropy) entropy[outIx] = H;
}

// Static device buffers reused across frames
static float* s_dEntropy  = nullptr;
static float* s_dContrast = nullptr;
static size_t s_tilesCap  = 0;

static bool colorize_ensure_metric_buffers(size_t tiles) {
    if (tiles <= s_tilesCap && s_dEntropy && s_dContrast) return true;

    if (s_dEntropy)  cudaFree(s_dEntropy);
    if (s_dContrast) cudaFree(s_dContrast);
    s_dEntropy = s_dContrast = nullptr;
    s_tilesCap = 0;

    cudaError_t rc = cudaMalloc((void**)&s_dEntropy,  tiles * sizeof(float));
    if (rc != cudaSuccess) return false;

    rc = cudaMalloc((void**)&s_dContrast, tiles * sizeof(float));
    if (rc != cudaSuccess) {
        cudaFree(s_dEntropy); s_dEntropy = nullptr;
        return false;
    }

    s_tilesCap = tiles;
    return true;
}

// Public API: compute metrics on GPU and copy to host vectors
bool colorize_compute_tile_metrics_to_host(
    const uint16_t* d_iterations,
    int width, int height, int tilePx,
    cudaStream_t stream,
    std::vector<float>& h_entropy,
    std::vector<float>& h_contrast) noexcept
{
    if (!d_iterations || width <= 0 || height <= 0 || tilePx <= 0) return false;

    const int px = (tilePx > 0) ? tilePx : 1;
    const int tilesX = (width  + px - 1) / px;
    const int tilesY = (height + px - 1) / px;
    const size_t tiles = (size_t)tilesX * (size_t)tilesY;

    if (!colorize_ensure_metric_buffers(tiles)) return false;

    dim3 grid((unsigned)tilesX, (unsigned)tilesY, 1);
    dim3 block(1, 1, 1);

    colorize_kernel_tile_metrics<<<grid, block, 0, stream>>>(
        d_iterations, width, height, px, tilesX, s_dEntropy, s_dContrast
    );
    if (cudaPeekAtLastError() != cudaSuccess) {
        return false;
    }

    h_entropy.resize(tiles);
    h_contrast.resize(tiles);

    cudaError_t rc = cudaMemcpyAsync(h_entropy.data(),  s_dEntropy,  tiles*sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(h_contrast.data(), s_dContrast, tiles*sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) {
        return false;
    }

    rc = cudaStreamSynchronize(stream);
    if (rc != cudaSuccess) {
        return false;
    }
    return true;
}
