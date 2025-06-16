# 🐭 Maus-Kommentar: Haupt-CMake-Datei für das Mandelbrot-Projekt (CUDA + OpenGL)

cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 🧠 Für IntelliSense und Clang-basierte Tools

# 🔧 CUDA separat aktivieren (Workaround für CMake 4.x)
enable_language(CUDA)
project(OtterDreamMandelbrot LANGUAGES CXX)

# ---------------------------------------------------------
# 📦 Vcpkg Integration
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# ---------------------------------------------------------
# ⚙️ Standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---------------------------------------------------------
# 🧠 CUDA-Architekturen
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(WARNING "CMAKE_CUDA_ARCHITECTURES ist nicht definiert. "
    "Idealerweise per Preset oder via -DCMAKE_CUDA_ARCHITECTURES gesetzt (z.B. 75;80;86;89)")
endif()

# ---------------------------------------------------------
# 📚 Pakete
find_package(OpenGL 4.3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

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

# 🧠 Precompiled Header
target_precompile_headers(mandelbrot_otterdream PRIVATE src/pch.hpp)

target_include_directories(mandelbrot_otterdream PRIVATE
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
)

set_target_properties(mandelbrot_otterdream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_features(mandelbrot_otterdream PRIVATE cxx_std_23)

target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /Zc:__cplusplus>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wno-unknown-pragmas>
    >
)

target_link_libraries(mandelbrot_otterdream PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart
    Threads::Threads
)

set_source_files_properties(
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/memory_utils.cu
    src/progressive.cu
    PROPERTIES LANGUAGE CUDA
)

# ---------------------------------------------------------
# 🧱 Build-Konfigurationen
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --fmad=false --prec-div=true")

# ---------------------------------------------------------
# 📦 Install-Ziel
install(TARGETS mandelbrot_otterdream RUNTIME DESTINATION bin)
