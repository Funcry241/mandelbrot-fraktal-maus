##### Otter: Alpha 81 - unified /MT CRT, static cudart, host-only hardening.
##### Schneefuchs: vcpkg triplet chosen by platform; hardening via HOST_LINK only; /WX consistent.
##### Maus: Multi-arch 80/86/89/90; consistent /MT incl. NVCC host; CUDA::cudart_static.

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90 CACHE STRING "Default architectures if none provided")
endif()

# vcpkg triplet - per platform
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
  if(WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows-static")
  else()
    set(VCPKG_TARGET_TRIPLET "x64-linux")
  endif()
endif()

project(mandelbrot_otterdream LANGUAGES CXX CUDA)

# vcpkg integration
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# Language standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Packages
find_package(OpenGL REQUIRED)
find_package(glew CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
# Boost.Multiprecision (header-only, clean via vcpkg/CONFIG)
find_package(Boost CONFIG REQUIRED COMPONENTS multiprecision)

# cuda_gl_interop.h location
find_path(CUDA_GL_INTEROP_INCLUDE_DIR cuda_gl_interop.h
    PATHS
        "$ENV{CUDA_PATH}/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
        "/usr/local/cuda/include"
        "/opt/cuda/include"
        "/usr/include"
    DOC "Path to cuda_gl_interop.h"
)
if(NOT CUDA_GL_INTEROP_INCLUDE_DIR)
  message(FATAL_ERROR "cuda_gl_interop.h not found - CUDA-OpenGL interop required")
endif()
include_directories(${CUDA_GL_INTEROP_INCLUDE_DIR})

# Executable: core renderer files
add_executable(mandelbrot_otterdream
    src/main.cpp
    src/luchs_log_host.cpp
    src/luchs_cuda_log_buffer.cu
    src/renderer_core.cu
    src/renderer_loop.cpp
    src/renderer_state.cpp
    src/renderer_pipeline.cpp
    src/renderer_window.cpp
    src/renderer_resources.cpp
    src/opengl_utils.cpp
    src/heatmap_overlay.cpp
    src/warzenschwein_overlay.cpp
    src/cuda_interop.cu
    src/core_kernel.cu
    src/zoom_logic.cpp
    src/zoom_command.cpp
    src/frame_pipeline.cpp
    src/frame_context.cpp
    src/hermelin_buffer.cpp
    src/bear_CudaPBOResource.cpp
    src/frame_capture.cpp
    src/fps_meter.cpp
    src/hud_text.cpp
    src/nacktmull.cu
    src/nacktmull_anchor.cpp
    src/nacktmull_engine.cpp
    src/nacktmull_shade.cu
    src/nacktmull_host.cpp
    src/progressive_iteration.cu
    src/progressive_shade_impl.cu
)

# Unify CRT to /MT so PCH and NVCC host compiler match (fixes LNK2038)
if(MSVC)
  set_property(TARGET mandelbrot_otterdream PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MT$<$<CONFIG:Debug>:d>>)
endif()

# Prefer static cudart to match /MT
target_link_libraries(mandelbrot_otterdream PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart_static
    Threads::Threads
    Boost::multiprecision
)

# GLEW compile definitions (avoid imaging/GLU)
target_compile_definitions(mandelbrot_otterdream PRIVATE
    GLEW_STATIC
    GLEW_NO_IMAGING
    GLEW_NO_GLU
)

# Precompiled headers (MSVC only)
if (MSVC)
  target_precompile_headers(mandelbrot_otterdream PRIVATE src/pch.hpp)
endif()

target_include_directories(mandelbrot_otterdream PRIVATE "${CMAKE_SOURCE_DIR}/src")

# Extra CUDA includes for IntelliSense (MSVC)
if (MSVC)
    list(APPEND extra_cuda_includes
        "$ENV{CUDA_PATH}/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/include"
    )
    foreach(path IN LISTS extra_cuda_includes)
        if(EXISTS "${path}/cuda_runtime.h")
            target_include_directories(mandelbrot_otterdream PRIVATE "${path}")
            break()
        endif()
    endforeach()
endif()

# CUDA separable compilation and device symbol resolution
set_target_properties(mandelbrot_otterdream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Explicit CUDA flags (do not leak linker/security flags into NVCC)
target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20 --expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /Zc:__cplusplus>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
    >
)

# Control Flow Guard only for C++ sources (host compile), not CUDA
target_compile_options(mandelbrot_otterdream PRIVATE
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/guard:cf>
)

# Mark CUDA source files explicitly
set_source_files_properties(
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/luchs_cuda_log_buffer.cu
    src/nacktmull.cu
    src/nacktmull_shade.cu
    src/progressive_iteration.cu
    src/progressive_shade_impl.cu
    PROPERTIES LANGUAGE CUDA
)

# Windows hardening in host link only (not NVCC device link/compile)
if(MSVC)
  # Release link optimizations: host link only
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<CONFIG:Release>:$<HOST_LINK:/INCREMENTAL:NO /OPT:REF /OPT:ICF>>
  )

  # Hardening flags: host link only (Release & RelWithDebInfo)
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<CONFIG:Release>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
    $<$<CONFIG:RelWithDebInfo>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
  )

  # Guard: require 64-bit toolchain
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "HIGHENTROPYVA requires 64-bit (/MACHINE:X64). Use a 64-bit toolchain.")
  endif()
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
