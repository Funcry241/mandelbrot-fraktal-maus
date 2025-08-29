##### MAUS: Alpha 81 — unified /MT CRT, static cudart, hardening host-only
# Datei: CMakeLists.txt
# 🐭 Maus-Kommentar: Alpha 81 – vcpkg Triplet dynamisch; **Windows-Hardening nur im HOST_LINK**.
# Fixes: konsistente /MT-CRT pro Target (inkl. PCH & NVCC-Host), `CUDA::cudart_static`, Multi-Arch (80/86/89/90).

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 🏗️ Architektur
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # 🐭 MAUS: Multi-Arch wie im bisherigen Build (80/86/89/90), statt nur 86.
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90 CACHE STRING "Default architectures if none provided")
endif()

# 🧩 Vcpkg Triplet – dynamisch nach Plattform
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
  if(WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows-static")
  else()
    set(VCPKG_TARGET_TRIPLET "x64-linux")
  endif()
endif()

project(mandelbrot_otterdream LANGUAGES CXX CUDA)

# ---------------------------------------------------------
# 🔧 Vcpkg Integration
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
set(VCPKG_MANIFEST_MODE ON)

# ---------------------------------------------------------
# ⚙️ Standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)  # CUDA23 wird nicht erkannt, explizit per compile_option unten
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---------------------------------------------------------
# 📚 Pakete
find_package(OpenGL REQUIRED)
find_package(glew CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
# ➕ Boost.Multiprecision (Header-only, aber sauber über vcpkg/CONFIG einbinden)
find_package(Boost CONFIG REQUIRED COMPONENTS multiprecision)

# ---------------------------------------------------------
# 🩹 Fix: Suche nach cuda_gl_interop.h
find_path(CUDA_GL_INTEROP_INCLUDE_DIR cuda_gl_interop.h
    PATHS
        "$ENV{CUDA_PATH}/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
        "/usr/local/cuda/include"
        "/opt/cuda/include"
        "/usr/include"  # Ubuntu-kompatible CUDA-Installationen
    DOC "Pfad zu cuda_gl_interop.h"
)

if(NOT CUDA_GL_INTEROP_INCLUDE_DIR)
  message(FATAL_ERROR "cuda_gl_interop.h not found - CUDA-OpenGL Interop benötigt!")
endif()

include_directories(${CUDA_GL_INTEROP_INCLUDE_DIR})

# ---------------------------------------------------------
# 🛠 Executable: zentrale Dateien des Mandelbrot-Renderers
add_executable(mandelbrot_otterdream
    src/main.cpp
    src/luchs_log_host.cpp
    src/luchs_cuda_log_buffer.cu
    src/renderer_core.cu
    src/renderer_loop.cpp
    src/renderer_state.cpp
    src/renderer_pipeline.cpp
    src/renderer_window.cpp
    src/renderer_resources.cpp
    src/opengl_utils.cpp
    src/heatmap_overlay.cpp
    src/warzenschwein_overlay.cpp
    src/cuda_interop.cu
    src/core_kernel.cu
    src/zoom_logic.cpp
    src/zoom_command.cpp
    src/frame_pipeline.cpp
    src/frame_context.cpp
    src/hermelin_buffer.cpp
    src/bear_CudaPBOResource.cpp
    src/frame_capture.cpp    
    src/fps_meter.cpp        
    src/hud_text.cpp    
    src/nacktmull.cu
    src/nacktmull_anchor.cpp
    src/nacktmull_engine.cpp
    src/nacktmull_shade.cu
    src/nacktmull_host.cpp
)

# MAUS: unify CRT to /MT so PCH and NVCC host compiler match (fixes LNK2038)
if(MSVC)
  set_property(TARGET mandelbrot_otterdream PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MT$<$<CONFIG:Debug>:d>>)
endif()

# MAUS: prefer static cudart to match /MT
target_link_libraries(mandelbrot_otterdream PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    CUDA::cudart_static
    Threads::Threads
    Boost::multiprecision
)

# 🛡️ GLEW: Imaging-Fehler verhindern durch explizite Compile-Definitionen
target_compile_definitions(mandelbrot_otterdream PRIVATE 
    GLEW_STATIC
    GLEW_NO_IMAGING
    GLEW_NO_GLU
)

# 🧠 PCH nur bei MSVC
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

# 🔧 CUDA-separat builden (erfordert Device-Link)
set_target_properties(mandelbrot_otterdream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON  # 🐭 MAUS: stabiler Device-Link über mehrere .cu-TUs
)

# ✅ explizit: CUDA bekommt --std=c++20, auch wenn CMake das Label nicht kennt
# (Keine Security-/Linker-Flags hier! Sonst landen sie via NVCC im Compile-/Device-Link.)
target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20 --expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /permissive- /Zc:__cplusplus>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
    >
)

# CFG (Control Flow Guard): nur C++-Quellen, nicht CUDA
# → verhindert, dass NVCC die Option sieht
#   (NVCC würde /guard:cf beim Device-Link nicht verstehen.)
target_compile_options(mandelbrot_otterdream PRIVATE
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/guard:cf>
)

# 🧠 CUDA-Dateien explizit markieren
set_source_files_properties(
    src/renderer_core.cu
    src/cuda_interop.cu
    src/core_kernel.cu
    src/luchs_cuda_log_buffer.cu
    src/nacktmull.cu
    src/nacktmull_shade.cu
    PROPERTIES LANGUAGE CUDA
)

# 🧱 Windows-Hardening NUR im Host-Link (nicht in NVCC Device-Link/Compile)
# Verhindert Fehler wie: "cl : D8021 Ungültiges numerisches Argument /HIGHENTROPYVA"
if(MSVC)
  # Release-Link-Optimierungen: nur Host-Link, nicht Device-Link
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<CONFIG:Release>:$<HOST_LINK:/INCREMENTAL:NO /OPT:REF /OPT:ICF>>
  )

  # Hardening-Flags: nur Host-Link (Release & RelWithDebInfo)
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<CONFIG:Release>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
    $<$<CONFIG:RelWithDebInfo>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
  )

  # Optional: saubere Fehlermeldung, falls 32-bit Toolchain erwischt wurde
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "HIGHENTROPYVA erfordert 64-bit (/MACHINE:X64). Bitte 64-bit-Toolchain verwenden.")
  endif()
endif()

# 🛠️ Build-Typ Default setzen
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
