///// Otter: Core kernel API surface minimized – classic path removed; use Capybara + Colorizer.
///// Schneefuchs: No CUDA types in public header; no hidden macros; headers/sources in sync; Settings is single source of truth (but not included here).
///// Maus: Stable typedefs only; dependents include capybara_frame_pipeline.cuh for rendering.
///// Datei: src/core_kernel.h

#pragma once

#include <cstdint>

// ============================================================================
// Core Kernel (lean, header-only API surface)
//
// The former classic render path (direct PBO write + separate E/C kernels) has
// been removed. The project now renders iterations via Capybara and visualizes
// through a dedicated colorizer:
//
//   Capybara → d_it (uint16) → colorize_iterations_to_pbo(...) → PBO
//
// Rendering entry points are provided via:
//   - capybara_selector.cuh   (launch_mandelbrot_select → Capybara)
//   - capybara_frame_pipeline.cuh (capy_render_and_analyze — now render-only)
//
// This header intentionally avoids CUDA headers to keep the public API clean.
// ============================================================================

namespace CoreKernel {

// Public, stable typedef for iteration storage (device buffers & host mirrors).
using IterationT = std::uint16_t;
static_assert(sizeof(IterationT) == 2, "IterationT must be 16-bit");

// Marker constants for compile-time checks in dependent code.
inline constexpr bool kClassicPboPathRemoved = true;

} // namespace CoreKernel
