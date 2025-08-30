///// Otter: OtterDream – CUDA/Host Coloring-Helpers (Smooth Iter, Paletten, Final Shade).
///// Schneefuchs: Header-only, zero deps; __host__/__device__ sicher; deterministisch; ASCII-only.
///// Maus: Keine Typ-Redefs außer Host-Fallback; Korrektur: nu = log2(log|z|) (kein Offset-Bug).

#pragma once

// Fallbacks, falls in Host-TUs inkludiert
#ifndef __CUDACC__
  #define __host__
  #define __device__
  #define __forceinline__
#endif

#ifdef __CUDACC__
  #include <cuda_runtime.h>
#else
  struct float3 { float x, y, z; };
  static inline float3 make_float3(float r, float g, float b){ return {r,g,b}; }
#endif

#include <cmath> // logf, log2f, powf, expf, sinf, floorf, fabsf

namespace otter {

// ---------- utils ----------
__host__ __device__ __forceinline__ float clamp01(float x){ return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
__host__ __device__ __forceinline__ float mix1(float a, float b, float t){ return a + (b - a) * t; }
__host__ __device__ __forceinline__ float3 mix3(const float3& a, const float3& b, float t){
  return make_float3(mix1(a.x,b.x,t), mix1(a.y,b.y,t), mix1(a.z,b.z,t));
}
__host__ __device__ __forceinline__ float3 mul3(const float3& c, float s){
  return make_float3(c.x*s, c.y*s, c.z*s);
}
__host__ __device__ __forceinline__ float3 add3(const float3& a, const float3& b){
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

// ---------- smooth iteration (continuous coloring) ----------
/*
  smoothIter:
  - Liefert kontinuierlichen Iterationswert (kein Banding) gemäß:
    nu = it + 1 - log2(log(|z|))  (klassische Douady–Hubbard-Form)
  - Rückgabe: < 0 => Pixel ist „im Set“ (nicht entkommen)
*/
__host__ __device__ __forceinline__ float smoothIter(int it, int maxIt, float zr, float zi){
  if (it >= maxIt) return -1.0f;
  // |z|^2:
  float r2 = zr*zr + zi*zi;
  // numerische Guards:
  if (r2 <= 0.f) r2 = 1e-30f;
  // log|z|:
  float log_zn = 0.5f * logf(r2);
  // Korrekt: nu = log2(log|z|) (ohne zusätzlichen Offset)
  float nu = log2f(fmaxf(log_zn, 1e-30f));
  float value = (float)it + 1.0f - nu;
  return value;
}

// ---------- palettes ----------
// Drei geschmackvolle Paletten (perzeptuell angenehm, UI-freundlich).
// 1) Aurora: tiefes Blau -> Cyan -> Türkis -> warmes Apricot
// 2) Glacier: Nachtblau -> Petrol -> Mint -> Elfenbein
// 3) Ember: Dunkelbraun -> Kupfer -> Orange -> Gelb (aber gedämpft)
enum class Palette : int { Aurora = 0, Glacier = 1, Ember = 2 };

// Keyframe-LUT (handverlesen, 7 Knoten je Palette)
__host__ __device__ __forceinline__ void paletteKey(const Palette p, int idx, float3& out){
  // Werte in [0,1]
  // Aurora
  static const float3 A[7] = {
    {0.050f,0.070f,0.150f}, // tiefblau
    {0.050f,0.240f,0.450f}, // blau
    {0.000f,0.450f,0.650f}, // cyan
    {0.000f,0.650f,0.620f}, // türkis
    {0.250f,0.780f,0.580f}, // jade
    {0.980f,0.780f,0.520f}, // sand
    {0.990f,0.860f,0.700f}  // warm apricot
  };
  // Glacier
  static const float3 G[7] = {
    {0.030f,0.040f,0.090f}, // night
    {0.035f,0.200f,0.300f}, // deep petrol
    {0.070f,0.350f,0.420f}, // petrol
    {0.120f,0.500f,0.520f}, // teal
    {0.300f,0.680f,0.620f}, // seafoam
    {0.700f,0.860f,0.780f}, // foam
    {0.950f,0.960f,0.910f}  // ivory
  };
  // Ember
  static const float3 E[7] = {
    {0.050f,0.030f,0.020f}, // near black, warm
    {0.150f,0.070f,0.030f}, // umber
    {0.320f,0.120f,0.040f}, // brown->copper
    {0.520f,0.230f,0.050f}, // copper
    {0.780f,0.340f,0.060f}, // amber
    {0.980f,0.520f,0.060f}, // orange
    {0.990f,0.820f,0.200f}  // golden
  };
  const float3* P = (p==Palette::Aurora) ? A : (p==Palette::Glacier ? G : E);
  out = P[idx];
}

// Takte t in [0,1] durch 7 Keyframes mit kubischer (Catmull-Rom) Interpolation.
__host__ __device__ __forceinline__ float3 palette(const Palette p, float t){
  t = clamp01(t);
  const int K = 7;
  float ft = t * (K - 1);
  int i = (int)ft;
  float u = ft - (float)i;
  // Nachbarn für Catmull-Rom holen (mit Randklemmen)
  int i0 = (i - 1 < 0)     ? 0     : i - 1;
  int i1 = i;
  int i2 = (i + 1 >= K)    ? K - 1 : i + 1;
  int i3 = (i + 2 >= K)    ? K - 1 : i + 2;

  float3 P0, P1, P2, P3;
  paletteKey(p, i0, P0);
  paletteKey(p, i1, P1);
  paletteKey(p, i2, P2);
  paletteKey(p, i3, P3);

  // Catmull-Rom Spline
  float u2 = u*u;
  float u3 = u2*u;

  auto cr = [&](float a, float b, float c, float d){
    return 0.5f * ( 2.f*b + (-a + c)*u + (2.f*a - 5.f*b + 4.f*c - d)*u2 + (-a + 3.f*b - 3.f*c + d)*u3 );
  };

  float3 R = make_float3(
    cr(P0.x, P1.x, P2.x, P3.x),
    cr(P0.y, P1.y, P2.y, P3.y),
    cr(P0.z, P1.z, P2.z, P3.z)
  );
  // leichte S-Kurve gegen flaches Mittelgrau
  float s = 0.15f;
  float m = (R.x + R.y + R.z) / 3.0f;
  R = mix3(R, make_float3(m,m,m), s*(0.5f - fabsf(0.5f - t)));
  // clamp
  R.x = clamp01(R.x); R.y = clamp01(R.y); R.z = clamp01(R.z);
  return R;
}

// ---------- final shading ----------
// Inputs:
//   - it, maxIt, zr, zi: Daten am Escape
//   - paletteId: Auswahl (Aurora/Glacier/Ember)
//   - stripeFreq/stripeAmp: dezentes Ring-Muster
//   - gamma: End-Gamma für Punch/Softness
// Rückgabe: float3 in [0,1]
__host__ __device__ __forceinline__
float3 shade(int it, int maxIt, float zr, float zi,
             Palette paletteId = Palette::Aurora,
             float stripeFreq = 3.0f,
             float stripeAmp  = 0.10f,
             float gamma      = 2.2f)
{
  float si = smoothIter(it, maxIt, zr, zi);
  if (si < 0.0f){
    // Mandelbrot-Inneres: edles tiefes Schwarz (leichtes Lift für Screens, aber sehr dunkel)
    return make_float3(0.02f, 0.02f, 0.03f);
  }
  // Normalisierung: „nu“ in [0,1] mit Softroll an den Enden
  float t = si / (float)maxIt;
  t = clamp01(t);
  // Kontrast-Stretch (logistic)
  const float k = 6.0f;                // Otter: ausgewogenes Punch
  t = 1.0f / (1.0f + expf(-k*(t - 0.5f)));

  // dezente Stripes (keine harten Zebra-Linien)
  const float PI = 3.14159265358979323846f;
  float stripes = 0.5f + 0.5f * sinf(2.0f*PI*stripeFreq*si);
  float stripeGain = 1.0f + stripeAmp * (stripes - 0.5f); // +/- Amp/2

  // Grundfarbe aus Palette
  float3 base = palette(paletteId, t);
  float3 col  = mul3(base, stripeGain);

  // Gamma-Korrektur
  float invG = 1.0f / gamma;
  col.x = powf(clamp01(col.x), invG);
  col.y = powf(clamp01(col.y), invG);
  col.z = powf(clamp01(col.z), invG);

  // sanfter Edge-Glow nahe Rand (ohne DE): nutze Fraktion von si als Proxy
  float frac = si - floorf(si);
  float glow = expf(-10.0f * fabsf(frac - 0.5f)); // Glöckchen um Halbinteger
  col = add3(col, mul3(make_float3(0.06f,0.06f,0.06f), glow));

  // clamp final
  col.x = clamp01(col.x); col.y = clamp01(col.y); col.z = clamp01(col.z);
  return col;
}

} // namespace otter
