cmake_minimum_required(VERSION 3.24)
project(mandelbrot_otterdream LANGUAGES CXX CUDA)

# ─── Manifest Mode für vcpkg ────────────────────────────────────────────────
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# ─── C++ & CUDA Standards ──────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── CUDA Setup ────────────────────────────────────────────────────────────
# Separable Compilation erlaubt Device-Linking über mehrere CU-Units
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# Ziel-Architekturen (ggf. an deine Karte anpassen)
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80)

# ─── OpenGL + Windowing über vcpkg ─────────────────────────────────────────
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)

# ─── Executable – alle .cu und .cpp Dateien ─────────────────────────────────
add_executable(mandelbrot_otterdream
    src/main.cu
    src/core_kernel.cu
)

# ─── Sprachstandards aktivieren ─────────────────────────────────────────────
target_compile_features(mandelbrot_otterdream PRIVATE
    cxx_std_17
    cuda_std_17
)

# ─── Link gegen Libraries ──────────────────────────────────────────────────
target_link_libraries(mandelbrot_otterdream
    PRIVATE
      # GLEW (Import-Library für glew32.dll)
      GLEW::GLEW
      # GLFW (Import-Library für glfw3.dll)
      glfw
      # OpenGL
      OpenGL::GL
)

# ─── Falls du GLEW statisch verwenden willst, entkommentiere: ──────────────
# target_compile_definitions(mandelbrot_otterdream PRIVATE GLEW_STATIC)

# ─── Optional: Mehr Warnungen und Debug-Infos ───────────────────────────────
# enable_testing()
# add_compile_options(-Wall -Wextra -Wpedantic)
# set(CMAKE_BUILD_TYPE Debug)
