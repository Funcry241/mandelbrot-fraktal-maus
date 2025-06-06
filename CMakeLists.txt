cmake_minimum_required(VERSION 3.24)
project(OtterDreamMandelbrot LANGUAGES CXX CUDA)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
set(VCPKG_MANIFEST_MODE ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

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

target_include_directories(mandelbrot_otterdream
  PRIVATE
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/include"
)

target_compile_options(mandelbrot_otterdream
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true --expt-relaxed-constexpr>
)

target_link_libraries(mandelbrot_otterdream
  PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart
)

target_compile_features(mandelbrot_otterdream
  PRIVATE
    cxx_std_20
    cuda_std_17
)

set_source_files_properties(
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/memory_utils.cu
    PROPERTIES
      LANGUAGE CUDA
)
