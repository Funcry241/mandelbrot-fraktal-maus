cmake_minimum_required(VERSION 3.24)
project(MandelbrotFraktal LANGUAGES CXX CUDA)

# ─── Manifest Mode für vcpkg ────────────────────────────────────────────────
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# ─── CUDA Setup ─────────────────────────────────────────────────────────────
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80)
set(CMAKE_CUDA_STANDARD 17)
# Separable Compilation einschalten (rdc)    
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ─── OpenGL + Windowing via vcpkg ────────────────────────────────────────────
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED CONFIG)
find_package(GLEW REQUIRED)

# ─── Executable mit allen Quell-Files ────────────────────────────────────────
# Wir listen hier direkt alle .cu/.cpp auf, anstatt eine separate library zu bauen.
add_executable(mandelbrot_otterdream
    src/main.cu
    src/core_kernel.cu      # enthält mandelbrotPersistent + computeComplexity
)

# separable compilation flag setzen
target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)

# ─── Link gegen Host-Libraries ──────────────────────────────────────────────
# Alle CUDA-Routinen werden automatisch von NVCC eingebunden, kein manuelles cudart-Target nötig.
target_link_libraries(mandelbrot_otterdream PRIVATE
    GLEW::GLEW
    glfw
    OpenGL::GL
)

# ─── Sprachstandards ─────────────────────────────────────────────────────────
target_compile_features(mandelbrot_otterdream PRIVATE
    cxx_std_17
    cuda_std_17
)
