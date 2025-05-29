# Datei: CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(MandelbrotFraktal LANGUAGES CXX CUDA)

# Manifest Mode
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# CUDA
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
enable_language(CUDA)

# Windowing & GL
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_library(cuda_core STATIC src/core_kernel.cu)
target_compile_options(cuda_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

add_executable(mandelbrot_otterdream src/main.cu)

target_link_libraries(mandelbrot_otterdream
    PRIVATE
      cuda_core
      GLEW::GLEW      # zuerst die Import-Lib für glew32.dll
      glfw            # dann GLFW
      OpenGL::GL      # und OpenGL
)

target_compile_features(mandelbrot_otterdream PRIVATE cxx_std_17 cuda_std_17)
