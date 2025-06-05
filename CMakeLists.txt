cmake_minimum_required(VERSION 3.24)
project(OtterDreamMandelbrot LANGUAGES CXX CUDA)

# -------------------------------------------------
# Vcpkg Toolchain, falls vorhanden
set(CMAKE_TOOLCHAIN_FILE
    "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")
set(VCPKG_MANIFEST_MODE ON)

# -------------------------------------------------
# Standards setzen
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD         17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------------------------
# CUDA Runtime dynamisch linken (keine doppelte cudart)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# -------------------------------------------------
# Wichtige Pakete
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED CONFIG)
find_package(GLEW REQUIRED)

# -------------------------------------------------
# Executable
add_executable(mandelbrot_otterdream
    src/main.cpp             # 🐭 jetzt Shader/VAO/VBO-Renderer
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/hud.cpp
    src/opengl_utils.cpp
    src/memory_utils.cu
    src/progressive.cpp
)

# -------------------------------------------------
# Include-Pfade
target_include_directories(mandelbrot_otterdream
  PRIVATE
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${VCPKG_INSTALLED_DIR}/x64-windows/include>
)

# -------------------------------------------------
# CUDA Compile Optionen
target_compile_options(mandelbrot_otterdream
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true --expt-relaxed-constexpr>
)

# -------------------------------------------------
# Link Libraries
target_link_libraries(mandelbrot_otterdream
  PRIVATE
    CUDA::cudart
    OpenGL::GL
    glfw
    GLEW::GLEW
)

# -------------------------------------------------
# Sprachfeatures aktivieren
target_compile_features(mandelbrot_otterdream
  PRIVATE
    cxx_std_20
    cuda_std_17
)

# -------------------------------------------------
# CUDA-Dateien deklarieren
set_source_files_properties(
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/memory_utils.cu
    PROPERTIES
      LANGUAGE CUDA
)

# -------------------------------------------------
# 🐭 Optional: Mehr Debug Infos
# set(CMAKE_VERBOSE_MAKEFILE ON)
