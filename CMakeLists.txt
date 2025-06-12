cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 🧠 Für IntelliSense und Clang-basierte Tools

project(OtterDreamMandelbrot LANGUAGES CXX CUDA)

# ---------------------------------------------------------
# 📦 Vcpkg Integration
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# ---------------------------------------------------------
# ⚙️ Standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20) # CUDA 12.9+: C++20 support (experimental but enabled)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Modern Target compile features
set(CMAKE_CXX_EXTENSIONS OFF) # No gnu++xx
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---------------------------------------------------------
# 📚 Pakete
find_package(OpenGL 4.3 REQUIRED)          # 🎯 OpenGL Mindestversion 4.3 wegen Core Profile
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)             # 🎯 Threads explizit

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
)

target_compile_features(mandelbrot_otterdream PRIVATE cxx_std_23)

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
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /Zc:__cplusplus>
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
    Threads::Threads
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
# 🎯 Präzisere CUDA Floating Point Math statt --use_fast_math
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --fmad=false --prec-div=true")

# ---------------------------------------------------------
# 📦 Install-Support
install(TARGETS mandelbrot_otterdream RUNTIME DESTINATION bin)
