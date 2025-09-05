##### Otter: Kill D9025 noise — remove any legacy /D GLEW_STATIC; no /U workaround needed; dynamic GLEW clean.
##### Schneefuchs: Static CUDA runtime enforced; /MT unified; separable compilation ON; host /WX; OpenGL::GLU + Boost explicit.
##### Maus: Arch 80/86/89/90; CUDA C++20; zero GLEW policy warnings; deterministic, duplicate-free defines.
##### Datei: CMakeLists.txt
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

# --- CUDA runtime policy: force STATIC everywhere --------------------------------
set(CMAKE_CUDA_USE_STATIC_CUDA_RUNTIME ON)   # legacy toggle still honored
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)       # modern toggle used by generators

# Language standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Packages
find_package(OpenGL REQUIRED)
find_package(glew CONFIG REQUIRED)           # vcpkg CONFIG package → GLEW::GLEW
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS container random system multiprecision)

# cuda_gl_interop.h location (interop required)
find_path(CUDA_GL_INTEROP_INCLUDE_DIR cuda_gl_interop.h
  PATHS
    "$ENV{CUDA_PATH}/include"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/include"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include"
    "/usr/local/cuda/include" "/opt/cuda/include" "/usr/include"
  DOC "Path to cuda_gl_interop.h"
)
if(NOT CUDA_GL_INTEROP_INCLUDE_DIR)
  message(FATAL_ERROR "cuda_gl_interop.h not found - CUDA-OpenGL interop required")
endif()
include_directories(${CUDA_GL_INTEROP_INCLUDE_DIR})

# --- Purge any legacy /D GLEW_STATIC coming from older config/caches -------------
# (Prevents MSVC D9025 '/DGLEW_STATIC' overridden by '/UGLEW_STATIC')
remove_definitions(-DGLEW_STATIC)
foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CUDA_FLAGS)
  if(DEFINED ${flag_var})
    string(REPLACE "/DGLEW_STATIC" "" ${flag_var} "${${flag_var}}")
    string(REPLACE "-DGLEW_STATIC" "" ${flag_var} "${${flag_var}}")
    set(${flag_var} "${${flag_var}}" CACHE STRING "" FORCE)
  endif()
endforeach()

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
  src/glfw_bootstrap.cpp
  src/async_frame_writer.cpp
  src/frame_budget_controller.cpp
)

# Unify CRT to /MT so PCH and NVCC host compiler match (fixes LNK2038)
if(MSVC)
  set_property(TARGET mandelbrot_otterdream PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MT$<$<CONFIG:Debug>:d>>)
endif()

# Link — prefer static cudart to match /MT (device runtime is expected with separable compilation)
target_link_libraries(mandelbrot_otterdream PRIVATE
  OpenGL::GL
  OpenGL::GLU
  GLEW::GLEW
  glfw
  CUDA::cudart_static
  Threads::Threads
  Boost::container
  Boost::random
  Boost::system
  Boost::multiprecision
)

# Target properties
set_target_properties(mandelbrot_otterdream PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Static
)

# Includes
target_include_directories(mandelbrot_otterdream PRIVATE "${CMAKE_SOURCE_DIR}/src")

# Explicit compiler options (no /U hack needed anymore)
target_compile_options(mandelbrot_otterdream PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20 --expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CXX>:
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /Zc:__cplusplus>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
  >
)

# GLEW: keep lean build flags (do NOT define GLEW_STATIC)
target_compile_definitions(mandelbrot_otterdream PRIVATE
  GLEW_NO_IMAGING
  GLEW_NO_GLU
)

# Control Flow Guard only for C++ sources (host compile), not CUDA
if(MSVC)
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/guard:cf>
  )
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<CONFIG:Release>:$<HOST_LINK:/INCREMENTAL:NO /OPT:REF /OPT:ICF>>
    $<$<CONFIG:Release>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
    $<$<CONFIG:RelWithDebInfo>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
  )
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "HIGHENTROPYVA requires 64-bit (/MACHINE:X64). Use a 64-bit toolchain.")
  endif()
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
