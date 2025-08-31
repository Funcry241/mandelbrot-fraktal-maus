///// Otter: CUDA-Guards – Host sieht keine CUDA-Builtins; Device nutzt logf/log2f & Grid-Indizes.
///// Schneefuchs: Deterministisch, ASCII-only; Header bleibt für .cpp-TUs kompilierbar.
///// Maus: Keine versteckten Pfade; Kernel nur unter __CUDACC__, Host bekommt nur die Launcher-Deklaration.
///// Datei: src/progressive_shade.cuh

#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cstdint>

namespace prog {

// Forward – Host/Device gemeinsam
class CudaProgressiveState;

// -----------------------------------------------------------------------------
// Device-only helpers & Kernel
// Sichtbar NUR, wenn der TU via NVCC kompiliert (__CUDACC__ definiert).
// Keine Direktnutzung interner CUDA-Header (keine math_functions.h).
// -----------------------------------------------------------------------------
#if defined(__CUDACC__)

  // --- Device helpers (header-only for inlining) ---
  static __device__ __forceinline__ float3 hsv_to_rgb(float h, float s, float v)
  {
      // h in [0,1), s,v in [0,1]
      float i = floorf(h * 6.0f);
      float f = h * 6.0f - i;
      float p = v * (1.0f - s);
      float q = v * (1.0f - f * s);
      float t = v * (1.0f - (1.0f - f) * s);
      int ii = ((int)i) % 6;
      float3 r;
      if      (ii == 0) r = make_float3(v, t, p);
      else if (ii == 1) r = make_float3(q, v, p);
      else if (ii == 2) r = make_float3(p, v, t);
      else if (ii == 3) r = make_float3(p, q, v);
      else if (ii == 4) r = make_float3(t, p, v);
      else              r = make_float3(v, p, q);
      return r;
  }

  static __device__ __forceinline__ uchar4 pack_rgb(float3 c)
  {
      c.x = fminf(fmaxf(c.x, 0.f), 1.f);
      c.y = fminf(fmaxf(c.y, 0.f), 1.f);
      c.z = fminf(fmaxf(c.z, 0.f), 1.f);
      return make_uchar4((unsigned char)(c.x * 255.0f + 0.5f),
                         (unsigned char)(c.y * 255.0f + 0.5f),
                         (unsigned char)(c.z * 255.0f + 0.5f),
                         255u);
  }

  static __device__ __forceinline__ float smooth_mu(uint32_t it, float2 z)
  {
      // Classic smooth coloring guard; |z| may be <= 1 early on
      const float r2 = z.x * z.x + z.y * z.y;
      if (r2 <= 1.0f) return (float)it;
      // mu = it + 1 - log2(log|z|)
      // Verwende logf/log2f (Device-Varianten), keine __logf/__log2f und keine internen Header.
      const float mu = (float)it + 1.0f - log2f(0.5f * logf(r2));
      return mu;
  }

  // --- Kernel: shade resume state to RGBA ---
  __global__ void k_shade_progressive_rgba(const uint8_t* __restrict__ flags,
                                           const uint32_t* __restrict__ it,
                                           const uint32_t* __restrict__ esc,
                                           const float2* __restrict__ z,
                                           uchar4* __restrict__ out,
                                           int width, int height,
                                           uint32_t maxIterCap)
  {
      const int x = blockIdx.x * blockDim.x + threadIdx.x;
      const int y = blockIdx.y * blockDim.y + threadIdx.y;
      if (x >= width || y >= height) return;
      const int i = y * width + x;

      const uint8_t  f    = flags[i];
      const uint32_t iter = it[i];

      float3 rgb;

      if (f & 0x01u) {
          // Escaped: smooth coloring over hue ring
          const float mu = smooth_mu(iter, z[i]);              // ~[0..maxIterCap]
          const float t  = fminf(mu / (float)maxIterCap, 1.0f);// normalize
          const float h  = fmodf(t * 0.85f, 1.0f);             // 0..~0.85
          rgb = hsv_to_rgb(h, 0.95f, 0.98f);
      } else if (f & 0x02u) {
          // Maxed (likely inside set): deep black
          rgb = make_float3(0.f, 0.f, 0.f);
      } else {
          // Survivor (noch nicht fertig): neutral preview (dim gray to cyan)
          const float t = fminf((float)iter / (float)maxIterCap, 1.0f);
          rgb = make_float3(0.15f + 0.3f * t, 0.2f + 0.6f * t, 0.25f + 0.7f * t);
      }

      out[i] = pack_rgb(rgb);
  }
#endif // __CUDACC__

// -----------------------------------------------------------------------------
// Host launcher – Deklaration bleibt für .cpp-TUs sichtbar.
// Die Definition erfolgt in einer .cu-Implementierungsdatei (z. B. progressive_shade_impl.cu).
// -----------------------------------------------------------------------------
void shade_progressive_to_rgba(const CudaProgressiveState& s,
                               uchar4* d_out,
                               int width, int height,
                               uint32_t maxIterCap,
                               cudaStream_t stream = 0);

} // namespace prog
