cmake_minimum_required(VERSION 3.24)
project(OtterDreamMandelbrot LANGUAGES CXX CUDA)

# -------------------------------------------------
#   vcpkg-Toolchain (falls Du vcpkg nutzt)
#   Diese Zeile nur aktivieren, wenn Du vcpkg eingebunden hast:
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")
set(VCPKG_MANIFEST_MODE ON)

# -------------------------------------------------
#   C++ & CUDA Standard
set(CMAKE_CXX_STANDARD    17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD   17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------------------------
#   Explizit auf dynamische CUDA-Runtime umschalten:
#   Verhindert, dass sowohl cudart.lib als auch cudart_static.lib eingebunden werden.
#   (Benötigt CMake ≥ 3.17.)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

#   Separables Compilation erlaubt device‐Code in mehreren .cu‐Dateien
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# -------------------------------------------------
#   Find necessary packages
find_package(CUDAToolkit REQUIRED)    # liefert CUDA::cudart
find_package(OpenGL REQUIRED)         # liefert OpenGL::GL
find_package(glfw3 3.3 REQUIRED CONFIG)  # GLFW::GLFW oder "glfw"
find_package(GLEW REQUIRED)           # liefert GLEW::GLEW

# -------------------------------------------------
#   Executable aus Deinen Quell‐Dateien
add_executable(mandelbrot_otterdream
    src/main.cu
    src/core_kernel.cu
)

#   Header‐Verzeichnis
target_include_directories(mandelbrot_otterdream
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)

# -------------------------------------------------
#   CUDA‐Compiler‐Optionen
#   -rdc=true aktiviert Separables Compilation
target_compile_options(mandelbrot_otterdream
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)

# -------------------------------------------------
#   Link gegen CUDA-Runtime (dynamisch), OpenGL, GLFW und GLEW
target_link_libraries(mandelbrot_otterdream
    PRIVATE
      CUDA::cudart
      OpenGL::GL
      glfw
      GLEW::GLEW
)

# -------------------------------------------------
#   Modern Language Features
target_compile_features(mandelbrot_otterdream
    PRIVATE
      cxx_std_17
      cuda_std_17
)
