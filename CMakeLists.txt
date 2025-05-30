cmake_minimum_required(VERSION 3.24)
project(MandelbrotFraktal LANGUAGES CXX CUDA)

# ─── Manifest Mode für vcpkg ────────────────────────────────────────────────
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# ─── CUDA Setup ─────────────────────────────────────────────────────────────
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ─── OpenGL + Windowing über vcpkg ──────────────────────────────────────────
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED CONFIG)
find_package(GLEW REQUIRED)

# (Remove any GLEW_STATIC define here!)

# ─── Executable inkl. aller CUDA-Units ──────────────────────────────────────
add_executable(mandelbrot_otterdream
    src/main.cu
    src/core_kernel.cu
)

# separable compilation flag setzen für alle CUDA-Units
target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)

# ─── Link gegen Host-Libraries ──────────────────────────────────────────────
#   Link dynamically against GLEW (the vcpkg import target),
#   GLFW and OpenGL.
target_link_libraries(mandelbrot_otterdream
    PRIVATE
      GLEW::GLEW
      glfw
      OpenGL::GL
)

# ─── Sprachstandards ─────────────────────────────────────────────────────────
target_compile_features(mandelbrot_otterdream PRIVATE
    cxx_std_17
    cuda_std_17
)
