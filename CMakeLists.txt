cmake_minimum_required(VERSION 3.24)
project(OtterDreamMandelbrot LANGUAGES CXX CUDA)

# -------------------------------------------------
# Falls ihr vcpkg nutzt, Toolchain‐Datei aktivieren:
set(CMAKE_TOOLCHAIN_FILE
    "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")
set(VCPKG_MANIFEST_MODE ON)

# -------------------------------------------------
# C++‐ und CUDA‐Standard setzen
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD         17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------------------------
# Dynamische CUDA-Runtime (Shared), damit nicht zweimal cudart.lib/cudart_static.lib gelinkt wird
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Separables Compilation (RDCC) erlauben
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# -------------------------------------------------
# Wichtige Pakete finden
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED CONFIG)
find_package(GLEW REQUIRED)

# -------------------------------------------------
# Executable bauen
add_executable(mandelbrot_otterdream
    src/main.cpp
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/hud.cpp
    src/opengl_utils.cpp
    src/memory_utils.cu
)

# -------------------------------------------------
# Include-Directories
target_include_directories(mandelbrot_otterdream
  BEFORE
  PRIVATE
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${VCPKG_INSTALLED_DIR}/x64-windows/include>
)

# -------------------------------------------------
# CUDA compile options
target_compile_options(mandelbrot_otterdream
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)

# -------------------------------------------------
# Linker
target_link_libraries(mandelbrot_otterdream
  PRIVATE
    CUDA::cudart
    OpenGL::GL
    glfw
    GLEW::GLEW
)

# -------------------------------------------------
# Moderne Sprachfeatures
target_compile_features(mandelbrot_otterdream
  PRIVATE
    cxx_std_20
    cuda_std_17
)

# -------------------------------------------------
# CUDA-Spezifikationen explizit setzen
set_source_files_properties(
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/memory_utils.cu        # ✅ HIER ERGÄNZT
    PROPERTIES
      LANGUAGE CUDA
)
