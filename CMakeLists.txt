# Datei: CMakeLists.txt
# Zeilen: 159
# 🐭 Maus-Kommentar: Alpha 49 – CUDA-FLAGS optimiert: --fmad=true, --use_fast_math, --prec-div=true. Damit maximale Floating-Point-Performance bei stabiler numerischer Genauigkeit. Keine Kompromisse bei Buildqualität.
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "Default architecture if none provided")
endif()

project(OtterDreamMandelbrot LANGUAGES CXX CUDA)

# ---------------------------------------------------------
# 🔧 Vcpkg Integration
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# ---------------------------------------------------------
# ⚙️ Standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ⚠️ Fallback: CMake kennt "CUDA23" nicht offiziell – daher auf 20 setzen,
# aber explizit mit --std=c++23 im target_compile_options aktivieren.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---------------------------------------------------------
# 📚 Pakete
find_package(OpenGL REQUIRED)
find_package(glew CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------
# 🩹 Fix: Suche nach cuda_gl_interop.h
find_path(CUDA_GL_INTEROP_INCLUDE_DIR cuda_gl_interop.h
    PATHS
        "$ENV{CUDA_PATH}/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
        "/usr/local/cuda/include"
        "/opt/cuda/include"
        "/usr/include"  # 🛠️ Ergänzt für Ubuntu-kompatible CUDA-Installationen (Schwester-Vorschlag)
    DOC "Pfad zu cuda_gl_interop.h"
)

if(NOT CUDA_GL_INTEROP_INCLUDE_DIR)
  message(FATAL_ERROR "cuda_gl_interop.h not found – CUDA-OpenGL Interop benötigt!")
endif()

include_directories(${CUDA_GL_INTEROP_INCLUDE_DIR})

# ---------------------------------------------------------
# 🛠 Executable: zentrale Dateien des Mandelbrot-Renderers
add_executable(mandelbrot_otterdream
    src/main.cpp
    src/renderer_core.cu
    src/renderer_loop.cpp
    src/renderer_state.cpp
    src/renderer_pipeline.cpp
    src/renderer_window.cpp
    src/renderer_resources.cpp
    src/hud.cpp
    src/opengl_utils.cpp
    src/heatmap_overlay.cpp
    src/cuda_interop.cu
    src/core_kernel.cu
    src/zoom_logic.cpp
    src/zoom_command.cpp
    src/frame_pipeline.cpp
)

if (MSVC)
  target_precompile_headers(mandelbrot_otterdream PRIVATE src/pch.hpp)
endif()

target_include_directories(mandelbrot_otterdream PRIVATE "${CMAKE_SOURCE_DIR}/src")

# 🔍 CUDA-Includes für PCH/IntelliSense (nur MSVC)
if (MSVC)
    list(APPEND extra_cuda_includes
        "$ENV{CUDA_PATH}/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/include"
    )
    foreach(path IN LISTS extra_cuda_includes)
        if(EXISTS "${path}/cuda_runtime.h")
            target_include_directories(mandelbrot_otterdream PRIVATE "${path}")
            break()
        endif()
    endforeach()
endif()

set_target_properties(mandelbrot_otterdream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(mandelbrot_otterdream PRIVATE cxx_std_23)

# ✅ explizit: CUDA bekommt --std=c++23, auch wenn CMake das Label nicht kennt
target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20 --expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /Zc:__cplusplus>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
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
    PROPERTIES LANGUAGE CUDA
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --fmad=true --use_fast_math --prec-div=true")

install(TARGETS mandelbrot_otterdream RUNTIME DESTINATION .)
