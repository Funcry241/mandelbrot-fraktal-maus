///// Otter: Core kernel TU – classic PBO + legacy E/C removed; Capybara is the only render path.
///// Schneefuchs: TU kept to preserve build structure; compile-time guards validate Settings.
///// Maus: No exports here; rendering flows through capybara_frame_pipeline + colorizer.
///// Datei: src/core_kernel.cu

#include "pch.hpp"
#include "settings.hpp"
#include "core_kernel.h"
#include <type_traits>

// ----------------------------------------------------------------------------
// Compile-time guards (fail early, deterministic).
// ----------------------------------------------------------------------------
static_assert(Settings::MANDEL_BLOCK_X > 0,  "MANDEL_BLOCK_X must be > 0");
static_assert(Settings::MANDEL_BLOCK_Y > 0,  "MANDEL_BLOCK_Y must be > 0");
static_assert((Settings::MANDEL_BLOCK_X % 32) == 0, "MANDEL_BLOCK_X must be a multiple of 32");

static_assert(Settings::MIN_TILE_SIZE  > 0,  "MIN_TILE_SIZE must be > 0");
static_assert(Settings::MAX_TILE_SIZE  >= Settings::MIN_TILE_SIZE, "MAX_TILE_SIZE must be >= MIN_TILE_SIZE");
static_assert(Settings::BASE_TILE_SIZE >= Settings::MIN_TILE_SIZE &&
              Settings::BASE_TILE_SIZE <= Settings::MAX_TILE_SIZE, "BASE_TILE_SIZE must be within [MIN, MAX]");

static_assert(Settings::MAX_ITERATIONS_CAP > 0, "MAX_ITERATIONS_CAP must be > 0");

static_assert(std::is_same_v<CoreKernel::IterationT, std::uint16_t>,
              "CoreKernel::IterationT must be uint16_t");

// ----------------------------------------------------------------------------
// Classic kernel/export removal statement
// ----------------------------------------------------------------------------
// The following symbols no longer exist and must not be referenced anywhere:
//   - launch_mandelbrotHybrid(...)
//   - computeCudaEntropyContrast(...)
// If you still see link errors referring to these, search and remove legacy
// includes/calls (see 'Specht' checklist).
//
// Rendering is now performed exclusively via:
//   - capy_render_and_analyze(...)   [capybara_frame_pipeline.cuh]
//   - colorize_iterations_to_pbo(...) [colorize_iterations.cuh]
//
// This TU intentionally exports no symbols.
// ----------------------------------------------------------------------------
