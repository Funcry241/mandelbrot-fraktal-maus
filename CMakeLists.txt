# Datei: CMakeLists.txt
# Zeilen: 154
# 🐭 Maus-Kommentar: Modularisierung + CUDA/OpenGL-Interop Fix. Ergänzt kompromisslose Warnungsbehandlung für MSVC/GCC/Clang/CUDA. Workaround für fehlende cuda_gl_interop.h – notwendig für Windows- und Linux-Setups. Schneefuchs: „Wer Fehler duldet, bekommt sie!“

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# Aktiviert die Ausgabe von compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Nur setzen, wenn nicht von außen definiert (z. B. Preset oder -D)
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
# 🩹 Fix: Suche nach cuda_gl_interop.h (wird nicht automatisch von CUDAToolkit eingebunden)
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
    # 🧠 Core Logic
    src/main.cpp
    src/renderer_core.cu
    src/renderer_loop.cpp
    src/renderer_state.cpp

    # 🎨 Rendering & UI
    src/renderer_pipeline.cpp
    src/renderer_window.cpp
    src/renderer_resources.cpp
    src/hud.cpp
    src/opengl_utils.cpp
    src/heatmap_overlay.cpp

    # ⚙️ CUDA-Kernel + Interop
    src/cuda_interop.cu
    src/core_kernel.cu

    # 🔎 Zoom-Logik & Analyse
    src/zoom_logic.cpp
    src/zoom_command.cpp
    src/frame_pipeline.cpp
)

# Precompiled Header (MSVC-only)
if (MSVC)
  target_precompile_headers(mandelbrot_otterdream PRIVATE src/pch.hpp)
endif()

# Includes
target_include_directories(mandelbrot_otterdream PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
)

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

# ✨ CUDA Eigenschaften
set_target_properties(mandelbrot_otterdream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Compile Flags: Kompromisslos!
target_compile_features(mandelbrot_otterdream PRIVATE cxx_std_23)

target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /Zc:__cplusplus>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
    >
)

# 🤝 Linking
target_link_libraries(mandelbrot_otterdream PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart
    Threads::Threads
)

# CUDA-Dateien explizit als solche markieren (für separable compilation)
set_source_files_properties(
    src/renderer_core.cu    
    src/cuda_interop.cu
    src/core_kernel.cu
    PROPERTIES LANGUAGE CUDA
)

# ---------------------------------------------------------
# 🛣️ Build-Typ
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --fmad=false --prec-div=true")

# ---------------------------------------------------------
# 📦 Zielpfad: installiere direkt in dist/, nicht dist/bin/
install(TARGETS mandelbrot_otterdream RUNTIME DESTINATION .)
