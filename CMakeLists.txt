cmake_minimum_required(VERSION 3.26)

project(OtterDreamMandelbrot LANGUAGES CXX CUDA)

# ---------------------------------------------------------
# 📦 Vcpkg Integration
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# ---------------------------------------------------------
# ⚙️ Standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------
# 📚 Pakete
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------
# 🛠️ Executable
add_executable(mandelbrot_otterdream
    src/main.cpp
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/hud.cpp
    src/opengl_utils.cpp
    src/memory_utils.cu
    src/progressive.cu
)

# ---------------------------------------------------------
# 📚 Include-Pfade
target_include_directories(mandelbrot_otterdream PRIVATE
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/include"
)

# ---------------------------------------------------------
# 🛠️ Target Properties
set_target_properties(mandelbrot_otterdream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_EXTENSIONS OFF
    CXX_STANDARD 23
    CUDA_STANDARD 17
)

# ---------------------------------------------------------
# 🧠 Dynamische GPU-Erkennung
set(DEFAULT_CUDA_ARCHITECTURES 75 80 86 89)
option(USE_NATIVE_GPU "Detect GPU architecture natively at build time" OFF)

if (USE_NATIVE_GPU)
    message(STATUS "Using native GPU architecture detection.")
    set_property(TARGET mandelbrot_otterdream PROPERTY CUDA_ARCHITECTURES native)
else()
    message(STATUS "Using default GPU architectures: ${DEFAULT_CUDA_ARCHITECTURES}")
    set_property(TARGET mandelbrot_otterdream PROPERTY CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCHITECTURES})
endif()

# ---------------------------------------------------------
# 🛡️ Compiler-Optionen
target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wno-unknown-pragmas>
    >
)

# ---------------------------------------------------------
# 🔗 Libraries
target_link_libraries(mandelbrot_otterdream PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart
)

# ---------------------------------------------------------
# 📜 CUDA-Dateien explizit als CUDA deklarieren
set_source_files_properties(
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/memory_utils.cu
    src/progressive.cu
    PROPERTIES LANGUAGE CUDA
)

# ---------------------------------------------------------
# 🧹 Build-Typen und Optimierungen
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# 🐭 Hinweis: --use_fast_math kann helfen, ist aber für numerisch extrem genaue Berechnungen manchmal problematisch.
