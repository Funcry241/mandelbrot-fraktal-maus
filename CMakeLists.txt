##### Otter: One-pass, vcpkg-friendly CMake with clean CUDA device-link (no host flags/libs leaking); robust GLFW/GLEW lookup with graceful fallbacks.
##### Schneefuchs: Host-only gating via $<NOT:$<DEVICE_LINK:1>> for link libs/options; no GLU headers; dynamic GLEW; CUDA13 guard; Ninja response files.
##### Maus: Deterministisch & schnell: keine Mehrfachläufe durch Toolchain-Fehlsuche; QUIET-find + Fallback; klare Diagnoseausgaben (ASCII).
##### Datei: CMakeLists.txt

cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# ---- Early: Multi-Config schlank & try_compile-Konfig ----
if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "Supported configs" FORCE)
endif()
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)

# Striktes Mapping "verwandter" Konfigurationen -> Release
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL     Release)
set(CMAKE_MAP_IMPORTED_CONFIG_RelWithDebInfo Release)
set(CMAKE_MAP_IMPORTED_CONFIG_MinSizeRel     Release)
set(CMAKE_MAP_IMPORTED_CONFIG_PROFILE        Release)
set(CMAKE_MAP_IMPORTED_CONFIG_CHECKED        Release)

# MSVC Runtime steuerbar
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(mandelbrot_otterdream LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------- Ninja QoL --------------------------
if(CMAKE_GENERATOR MATCHES "Ninja")
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON)
  set_property(GLOBAL PROPERTY JOB_POOLS console_pool=1 link_pool=1)
  set(CMAKE_JOB_POOL_LINK link_pool)
endif()

# ---------------------- Architectures ----------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90 CACHE STRING "Default architectures if none provided")
endif()

# ---------------------- vcpkg prefix (Fallbacks) -----------
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
  if(WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows")
  else()
    set(VCPKG_TARGET_TRIPLET "x64-linux")
  endif()
endif()
if(NOT DEFINED VCPKG_INSTALLED_DIR)
  if(DEFINED ENV{VCPKG_INSTALLED_DIR})
    set(VCPKG_INSTALLED_DIR "$ENV{VCPKG_INSTALLED_DIR}")
  else()
    set(VCPKG_INSTALLED_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed")
  endif()
endif()
set(_VCPKG_PREFIX "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

# ---------------------- Language Std -----------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---------------------- Packages ---------------------------
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Platform OpenGL lib alias
if(WIN32)
  set(_OPENGL_GL_LIB opengl32)
else()
  set(_OPENGL_GL_LIB OpenGL::GL)
endif()

# --- GLFW (robust: QUIET + Fallback) -----------------------
set(GLFW_LINK_ITEM "")
# 1) Prefer config package (vcpkg toolchain provides this)
find_package(glfw3 CONFIG QUIET)
if(TARGET glfw)
  set(GLFW_LINK_ITEM glfw)
elseif(TARGET glfw3)
  set(GLFW_LINK_ITEM glfw3)
else()
  # 2) Fallback to raw lib search (works even w/o toolchain)
  find_library(GLFW_LINK_ITEM
    NAMES glfw3 glfw3dll glfw
    HINTS "${_VCPKG_PREFIX}/lib" "${_VCPKG_PREFIX}/debug/lib"
  )
  if(NOT GLFW_LINK_ITEM)
    message(FATAL_ERROR "GLFW not found. Either use the vcpkg toolchain (-DCMAKE_TOOLCHAIN_FILE=.../vcpkg.cmake) or install glfw3.")
  endif()
endif()

# --- GLEW (dynamic, no GLU) --------------------------------
set(GLEW_USE_STATIC_LIBS OFF CACHE BOOL "Link GLEW dynamically" FORCE)
set(GLEW_LINK_ITEM "")
find_package(GLEW CONFIG QUIET)
if(TARGET GLEW::GLEW)
  set(GLEW_LINK_ITEM GLEW::GLEW)
elseif(TARGET GLEW::glew)
  set(GLEW_LINK_ITEM GLEW::glew)
elseif(TARGET GLEW::glew32)
  set(GLEW_LINK_ITEM GLEW::glew32)
else()
  find_library(GLEW_LINK_ITEM
    NAMES glew32 glew GLEW
    HINTS "${_VCPKG_PREFIX}/lib" "${_VCPKG_PREFIX}/debug/lib"
  )
  if(NOT GLEW_LINK_ITEM)
    message(FATAL_ERROR "GLEW not found. Either use the vcpkg toolchain (-DCMAKE_TOOLCHAIN_FILE=.../vcpkg.cmake) or install glew.")
  endif()
endif()

# ---------------------- Feature-Schalter -------------------
option(OTTER_USE_ORT           "Enable ONNX Runtime AOP (Policy-Replikatoren)" OFF)
option(OTTER_USE_NVRTC         "Enable NVRTC Coloring (Color-Replikatoren)"   OFF)
option(OTTER_LINK_CUDA_DRIVER  "Link the CUDA Driver API (cu*). Enable only if you call cu* symbols." OFF)
option(OTTER_LINK_VERBOSE      "Add /VERBOSE:LIB to MSVC link for diagnostics (host-only)" ON)

# ---------------------- Executable -------------------------
add_executable(mandelbrot_otterdream
  src/main.cpp
  src/luchs_log_host.cpp
  src/luchs_cuda_log_buffer.cu
  src/renderer_core.cu
  src/renderer_loop.cpp
  src/renderer_pipeline.cpp
  src/renderer_window.cpp
  src/renderer_resources.cpp
  src/heatmap_overlay.cpp
  src/hud_text.cpp
  src/warzenschwein_overlay.cpp
  src/cuda_interop.cu
  # Capybara path
  src/capybara_render_kernel.cu
  src/colorize_iterations.cu
  # Zoom & control
  src/zoom_logic.cpp
  # Frame + context + utilities
  src/frame_pipeline.cpp
  src/frame_context.cpp
  src/hermelin_buffer.cpp
  src/bear_CudaPBOResource.cpp
  src/fps_meter.cpp
  src/renderer_state_cuda.cpp
  src/renderer_state_gl.cpp
  src/heatmap_metrics.cu
  # Axolotel HUD (WOW overlay)
  src/axolotel_hud.cpp
  src/axolotel_coupler.cpp
  # Dachs HUD (tri-pane top layout)
  src/dachs_hud.cpp
  # ---------- Replikatoren (core; optionale werden unten ergänzt) ----------
  src/perturb_core.cpp
  src/perturb_orbit_mp.cpp
  src/luchs_dsl.cpp
)

# Sicherstellen, dass Linker-Sprache korrekt erkannt wird
set_property(TARGET mandelbrot_otterdream PROPERTY LINKER_LANGUAGE CXX)

# ---------- Optionale Replikatoren-Quellen ----------
if(OTTER_USE_ORT)
  target_sources(mandelbrot_otterdream PRIVATE
    src/ai/onnx_model.cpp
    src/ai/aop_controller.cpp
  )
endif()
if(OTTER_USE_NVRTC)
  target_sources(mandelbrot_otterdream PRIVATE
    src/coloring_runtime_nvrtc.cpp
  )
endif()

# --- MSVC: gezielt C4702 nur für perturb_core.cpp ----------
if(MSVC)
  set_source_files_properties(
    src/perturb_core.cpp
    PROPERTIES COMPILE_OPTIONS "/wd4702"
  )
endif()

# ---------------------- Compile-Options --------------------
if(MSVC)
  set_property(TARGET mandelbrot_otterdream PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /sdl /permissive- /Zc:__cplusplus /EHsc /Brepro>
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Xcompiler=/W4
      -Xcompiler=/WX
      -Xcompiler=/diagnostics:caret
      -Xcompiler=/permissive-
      -Xcompiler=/Zc:__cplusplus
      -Xcompiler=/EHsc
      -Xcompiler=/wd4505
    >
  )
else()
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror -Wno-unknown-pragmas>
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Xcompiler=-Wall
      -Xcompiler=-Wextra
      -Xcompiler=-Wpedantic
      -Xcompiler=-Werror
    >
  )
endif()

# CUDA device flags
target_compile_options(mandelbrot_otterdream PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20 --expt-relaxed-constexpr>
)

# ---------------------- Target-Properties ------------------
set_target_properties(mandelbrot_otterdream PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Shared
)

# ---------------------- Includes ---------------------------
target_include_directories(mandelbrot_otterdream PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  ${CUDAToolkit_INCLUDE_DIRS}
  "${_VCPKG_PREFIX}/include"   # Fallback-Includes wenn find_package nicht alle Targets setzt
)

# ---------------------- Defines ----------------------------
target_compile_definitions(mandelbrot_otterdream PRIVATE
  GLEW_NO_IMAGING
  GLEW_NO_GLU
  CAPY_DEFAULT_ON=1
  $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX WIN32_LEAN_AND_MEAN>
)
target_compile_definitions(mandelbrot_otterdream PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:__NV_NO_VECTOR_DEPRECATION_DIAG>
  $<$<BOOL:${OTTER_USE_ORT}>:OTTER_USE_ORT=1>
  $<$<BOOL:${OTTER_USE_NVRTC}>:OTTER_USE_NVRTC=1>
)

# ---------------------- Link: Host-only gating -------------
# Wichtig: Keine Host-Libs/-Optionen in die NVCC-Device-Link-Phase durchreichen.
target_link_libraries(mandelbrot_otterdream PRIVATE
  $<$<NOT:$<DEVICE_LINK:1>>:${_OPENGL_GL_LIB}>
  $<$<NOT:$<DEVICE_LINK:1>>:${GLEW_LINK_ITEM}>
  $<$<NOT:$<DEVICE_LINK:1>>:${GLFW_LINK_ITEM}>
  $<$<NOT:$<DEVICE_LINK:1>>:Threads::Threads>
  $<$<AND:$<NOT:$<DEVICE_LINK:1>>,$<BOOL:WIN32>>:user32 gdi32 shell32 advapi32>
)

# ---------------------- Link-Optionen (host-only) ----------
if(MSVC)
  target_link_options(mandelbrot_otterdream PRIVATE
    # Diagnose optional
    $<$<AND:$<NOT:$<DEVICE_LINK:1>>,$<BOOL:${OTTER_LINK_VERBOSE}>>:/VERBOSE:LIB>

    # Sicherheits- & Repro-Flags
    $<$<NOT:$<DEVICE_LINK:1>>:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf /Brepro>

    # Release-Optimierungen
    $<$<AND:$<NOT:$<DEVICE_LINK:1>>,$<CONFIG:Release>>:/INCREMENTAL:NO /OPT:REF /OPT:ICF>

    # GLU hart ausschließen – nur Hostlink
    $<$<NOT:$<DEVICE_LINK:1>>:/NODEFAULTLIB:GLU32 /NODEFAULTLIB:glu32>
  )

  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "HIGHENTROPYVA erfordert 64-bit (/MACHINE:X64). Bitte 64-bit Toolchain verwenden.")
  endif()
else()
  # Linux: -pthread nur beim Host-Link
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<NOT:$<DEVICE_LINK:1>>:-pthread>
  )
endif()

# --- CUDA Driver API (optional, host-only) -----------------
if(OTTER_LINK_CUDA_DRIVER)
  set(_USE_CUDA_DRIVER_TARGET OFF)
  if(TARGET CUDA::cuda_driver)
    foreach(_cfg IN ITEMS RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
      get_target_property(_drv_loc CUDA::cuda_driver IMPORTED_LOCATION_${_cfg})
      if(_drv_loc)
        set(_USE_CUDA_DRIVER_TARGET ON)
      endif()
    endforeach()
  endif()

  if(_USE_CUDA_DRIVER_TARGET)
    target_link_libraries(mandelbrot_otterdream PRIVATE
      $<$<NOT:$<DEVICE_LINK:1>>:CUDA::cuda_driver>)
  else()
    find_library(CUDA_DRIVER_LIB
      NAMES cuda nvcuda
      HINTS
        "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_LIBRARY_DIR}/x64"
        "$ENV{CUDA_PATH}/lib/x64"
        "$ENV{SystemRoot}/System32"
    )
    if(CUDA_DRIVER_LIB)
      target_link_libraries(mandelbrot_otterdream PRIVATE
        $<$<NOT:$<DEVICE_LINK:1>>:${CUDA_DRIVER_LIB}>)
    else()
      message(STATUS "CUDA Driver API nicht gefunden – überspringe Verlinkung (cu* ungenutzt).")
    endif()
  endif()
endif()

# --- AppLocal: Runtime-DLLs neben die EXE kopieren (host-only) ---
add_custom_command(TARGET mandelbrot_otterdream POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_RUNTIME_DLLS:mandelbrot_otterdream>
          $<TARGET_FILE_DIR:mandelbrot_otterdream>
  COMMAND_EXPAND_LISTS
)

# ---------------------- CUDA 13 Guard ----------------------
if(CUDAToolkit_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "CUDA 13.0 oder neuer ist erforderlich. Gefunden: ${CUDAToolkit_VERSION}")
endif()
find_file(CUDA_GL_INTEROP_HEADER cuda_gl_interop.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
if(NOT CUDA_GL_INTEROP_HEADER)
  message(FATAL_ERROR "cuda_gl_interop.h nicht im CUDAToolkit gefunden.")
endif()

# ---------------------- Defaults (Single-Config) -----------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------- Diagnose ---------------------------
message(STATUS "Otter Configs: ${CMAKE_CONFIGURATION_TYPES}")
message(STATUS "Map RelWithDebInfo -> ${CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO}")
get_target_property(_OTTER_LINK_LANG mandelbrot_otterdream LINKER_LANGUAGE)
message(STATUS "Otter Linker language: ${_OTTER_LINK_LANG}")
