##### Otter: Phase-1 Replikatoren aktiv (Stubs/Logs), Build bleibt ohne ORT/NVRTC-Link
##### Schneefuchs: Dynamisches GLEW; **harter Anti-GLU-Filter** (auch Pfad-Tokens & importierte Targets)
##### Maus: ASCII-only; configure-Echo der OTTER_USE_*-Schalter; /WX-safe unverändert
##### Datei: CMakeLists.txt
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(mandelbrot_otterdream LANGUAGES CXX CUDA)

# Allow MSVC runtime to be controlled by MSVC_RUNTIME_LIBRARY
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Map imported RELWITHDEBINFO -> Release for imported targets that don't provide it
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)

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

# ---------------------- vcpkg Triplet ----------------------
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
  if(WIN32)
    set(VCPKG_TARGET_TRIPLET "x64-windows")
  else()
    set(VCPKG_TARGET_TRIPLET "x64-linux")
  endif()
endif()

# Resolve vcpkg installed dir for manual fallbacks
if(NOT DEFINED VCPKG_INSTALLED_DIR)
  if(DEFINED ENV{VCPKG_INSTALLED_DIR})
    set(VCPKG_INSTALLED_DIR "$ENV{VCPKG_INSTALLED_DIR}")
  else()
    set(VCPKG_INSTALLED_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed")
  endif()
endif()
set(_VCPKG_PREFIX "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

# ---------------------- Language Std -----------------------
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ---------------------- Feature Toggles (safe defaults OFF) ----------
option(OTTER_USE_NVRTC "Enable NVRTC coloring runtime (links provided by CUDA Toolkit)" OFF)
option(OTTER_USE_ORT   "Enable ONNX Runtime integration (expects ORT in toolchain)" OFF)

# ORT wiring (prereqs only; default OFF keeps build without ORT link)
set(OTTER_ORT_ROOT "" CACHE PATH "Root directory of ONNX Runtime (optional if CMake package is available)")
set(OTTER_ORT_PROVIDER "cuda" CACHE STRING "ONNX Runtime execution provider: cpu|cuda")
set_property(CACHE OTTER_ORT_PROVIDER PROPERTY STRINGS cpu cuda)

# Configure-time echo for visibility (Phase-1: both OFF is expected)
message(STATUS "OTTER_USE_NVRTC = ${OTTER_USE_NVRTC}")
message(STATUS "OTTER_USE_ORT   = ${OTTER_USE_ORT}")
message(STATUS "OTTER_ORT_ROOT  = ${OTTER_ORT_ROOT}")
message(STATUS "OTTER_ORT_PROVIDER = ${OTTER_ORT_PROVIDER}")

# ---------------------- Packages ---------------------------
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)                 # OpenGL::GL
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# --- GLFW (prefer imported target, else direct import-lib) ----
set(GLFW_LINK_ITEM "")
if(WIN32)
  find_package(glfw3 CONFIG REQUIRED)         # usually provides 'glfw'
  if(TARGET glfw)
    set(GLFW_LINK_ITEM glfw)
  elseif(TARGET glfw3)
    set(GLFW_LINK_ITEM glfw3)
  else()
    find_library(GLFW_LINK_ITEM
      NAMES glfw3 glfw3dll glfw
      HINTS "${_VCPKG_PREFIX}/lib" "${_VCPKG_PREFIX}/debug/lib"
      REQUIRED)
  endif()
else()
  find_package(glfw3 CONFIG REQUIRED)
  set(GLFW_LINK_ITEM glfw)
endif()

# --- GLEW (dynamic, no GLU; robust mapping) ----------------
set(GLEW_USE_STATIC_LIBS OFF CACHE BOOL "Link GLEW dynamically" FORCE)
remove_definitions(-DGLEW_STATIC)

set(GLEW_LINK_ITEM "")
if(WIN32)
  find_package(GLEW CONFIG REQUIRED)
  if(TARGET GLEW::GLEW)
    set(GLEW_LINK_ITEM GLEW::GLEW)
  elseif(TARGET GLEW::glew)
    set(GLEW_LINK_ITEM GLEW::glew)
  elseif(TARGET GLEW::glew32)
    set(GLEW_LINK_ITEM GLEW::glew32)
  else()
    find_library(GLEW_LINK_ITEM
      NAMES glew32 glew
      HINTS "${_VCPKG_PREFIX}/lib" "${_VCPKG_PREFIX}/debug/lib"
      REQUIRED)
  endif()
else()
  find_package(GLEW REQUIRED)
  if(TARGET GLEW::GLEW)
    set(GLEW_LINK_ITEM GLEW::GLEW)
  else()
    find_library(GLEW_LINK_ITEM NAMES GLEW glew REQUIRED)
  endif()
endif()

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
  # --- Replikatoren / Orbit & Color ---
  src/perturb_core.cpp
  src/perturb_orbit_mp.cpp
  src/luchs_dsl.cpp
  src/coloring_runtime_nvrtc.cpp
  # --- AI (AOP/Policy) now compiled-in (safe even without ORT)
  src/ai/aop_controller.cpp
  src/ai/onnx_model.cpp
)

# /WX locker nur für onnx_model.cpp (Third-Party-Headers etc.)
set_property(SOURCE src/ai/onnx_model.cpp
  APPEND PROPERTY COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:MSVC>:/WX->"
    "$<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:CXX>>:-Wno-error>"
)

# ---------------------- Compile-Options --------------------
if(MSVC)
  set_property(TARGET mandelbrot_otterdream PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  target_compile_options(mandelbrot_otterdream PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /sdl /permissive- /Zc:__cplusplus /EHsc /Brepro /MP>
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
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Xcompiler=-Wall
      -Xcompiler=-Wextra
      -Xcompiler=-Wpedantic
      -Xcompiler=-Werror
    >
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mandelbrot_otterdream PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
    )
  endif()
endif()

# CUDA device flags (pure device)
target_compile_options(mandelbrot_otterdream PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20 --expt-relaxed-constexpr>
)

# ---------------------- Target-Properties ------------------
set_target_properties(mandelbrot_otterdream PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Shared   # align with /MD to avoid LIBCMT conflict
)

# ---------------------- Includes ---------------------------
target_include_directories(mandelbrot_otterdream PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  ${CUDAToolkit_INCLUDE_DIRS}
)

# ---------------------- Defines ----------------------------
target_compile_definitions(mandelbrot_otterdream PRIVATE
  GLEW_NO_IMAGING
  GLEW_NO_GLU
  CAPY_DEFAULT_ON=1
  # Windows headers lean & no min/max macros
  $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX WIN32_LEAN_AND_MEAN>
)
target_compile_definitions(mandelbrot_otterdream PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:__NV_NO_VECTOR_DEPRECATION_DIAG>
)

# Optional feature defines (compile-time gates; linking left to toolchain/user)
if(OTTER_USE_NVRTC)
  target_compile_definitions(mandelbrot_otterdream PRIVATE OTTER_USE_NVRTC=1)
endif()
if(OTTER_USE_ORT)
  target_compile_definitions(mandelbrot_otterdream PRIVATE OTTER_USE_ORT=1)
endif()

# ---------------------- Link -------------------------------
target_link_libraries(mandelbrot_otterdream PRIVATE
  OpenGL::GL
  ${GLEW_LINK_ITEM}
  ${GLFW_LINK_ITEM}
  # Do NOT link CUDA::cudart manually — CMake wires cudart for CUDA targets;
  # explicit linking breaks on RelWithDebInfo when that config isn't exported.
  $<$<TARGET_EXISTS:CUDA::cudadevrt>:CUDA::cudadevrt>
  Threads::Threads
)

# NVRTC (only when explicitly enabled)
if(OTTER_USE_NVRTC)
  if(TARGET CUDA::nvrtc)
    target_link_libraries(mandelbrot_otterdream PRIVATE CUDA::nvrtc)
  else()
    message(FATAL_ERROR "OTTER_USE_NVRTC=ON, but CUDA::nvrtc target not found (install CUDA Toolkit with NVRTC).")
  endif()
  # CUDA driver is required for module loading/launch via driver API
  if(TARGET CUDA::cuda_driver)
    target_link_libraries(mandelbrot_otterdream PRIVATE CUDA::cuda_driver)
  else()
    message(FATAL_ERROR "OTTER_USE_NVRTC=ON, but CUDA::cuda_driver target not found.")
  endif()
endif()

# ORT (robust conditional wiring; default OFF keeps build w/o ORT)
if(OTTER_USE_ORT)
  set(_OTTER_ORT_TGT "")
  # 1) Try a CMake package that exports targets
  find_package(onnxruntime CONFIG QUIET)
  if(TARGET onnxruntime)
    set(_OTTER_ORT_TGT onnxruntime)
  elseif(TARGET onnxruntime::onnxruntime)
    set(_OTTER_ORT_TGT onnxruntime::onnxruntime)
  endif()

  # 2) Fallback to manual find if no target yet and a root is provided
  if(NOT _OTTER_ORT_TGT)
    if(OTTER_ORT_ROOT)
      find_path(_OTTER_ORT_INC onnxruntime_c_api.h
        HINTS "${OTTER_ORT_ROOT}/include" "${OTTER_ORT_ROOT}/include/onnxruntime" REQUIRED)
      find_library(_OTTER_ORT_LIB onnxruntime
        HINTS "${OTTER_ORT_ROOT}/lib" "${OTTER_ORT_ROOT}/lib64" REQUIRED)
      target_include_directories(mandelbrot_otterdream PRIVATE "${_OTTER_ORT_INC}")
      set(_OTTER_ORT_TGT "${_OTTER_ORT_LIB}")
    else()
      message(FATAL_ERROR "OTTER_USE_ORT=ON, aber kein onnxruntime CMake-Paket und kein OTTER_ORT_ROOT angegeben.")
    endif()
  endif()

  # Link core ORT
  target_link_libraries(mandelbrot_otterdream PRIVATE ${_OTTER_ORT_TGT})

  # Optional CUDA provider
  if(OTTER_ORT_PROVIDER STREQUAL "cuda")
    if(TARGET onnxruntime_providers_cuda)
      target_link_libraries(mandelbrot_otterdream PRIVATE onnxruntime_providers_cuda)
    else()
      find_library(_OTTER_ORT_CUDA onnxruntime_providers_cuda
        HINTS "${OTTER_ORT_ROOT}/lib" "${OTTER_ORT_ROOT}/lib64")
      if(_OTTER_ORT_CUDA)
        target_link_libraries(mandelbrot_otterdream PRIVATE "${_OTTER_ORT_CUDA}")
      else()
        message(FATAL_ERROR "OTTER_ORT_PROVIDER=cuda gewählt, aber 'onnxruntime_providers_cuda' nicht gefunden.")
      endif()
    endif()
  endif()
endif()

# --- Anti-GLU Filter für Windows (erzwingt "no GLU") ------------------------
if(WIN32)
  # 1) Strip GLU from this target's link interface (handles absolute/relative tokens).
  get_target_property(_OTTER_LINK_LIBS mandelbrot_otterdream LINK_LIBRARIES)
  if(_OTTER_LINK_LIBS)
    set(_OTTER_LINK_LIBS_FILTERED "${_OTTER_LINK_LIBS}")
    # Match elements that end with 'glu32' or 'glu32.lib' (case-insensitive), possibly with a path.
    list(FILTER _OTTER_LINK_LIBS_FILTERED EXCLUDE REGEX "([/\\\\]|^)[Gg][Ll][Uu]32(\\.lib)?$")
    list(FILTER _OTTER_LINK_LIBS_FILTERED EXCLUDE REGEX "OpenGL::GLU")
    if(NOT _OTTER_LINK_LIBS_FILTERED STREQUAL "${_OTTER_LINK_LIBS}")
      set_property(TARGET mandelbrot_otterdream PROPERTY LINK_LIBRARIES "${_OTTER_LINK_LIBS_FILTERED}")
    endif()
  endif()

  # 2) Patch common GLEW imported targets if they transitively add GLU.
  foreach(_glew_tgt IN ITEMS GLEW::GLEW GLEW::glew GLEW::glew32)
    if(TARGET ${_glew_tgt})
      get_target_property(_GLEW_LIBS ${_glew_tgt} INTERFACE_LINK_LIBRARIES)
      if(_GLEW_LIBS)
        set(_GLEW_LIBS_FILTERED "${_GLEW_LIBS}")
        list(FILTER _GLEW_LIBS_FILTERED EXCLUDE REGEX "([/\\\\]|^)[Gg][Ll][Uu]32(\\.lib)?$")
        list(FILTER _GLEW_LIBS_FILTERED EXCLUDE REGEX "OpenGL::GLU")
        if(NOT _GLEW_LIBS_FILTERED STREQUAL "${_GLEW_LIBS}")
          set_property(TARGET ${_glew_tgt} PROPERTY INTERFACE_LINK_LIBRARIES "${_GLEW_LIBS_FILTERED}")
        endif()
      endif()
    endif()
  endforeach()
endif()

# Optional: /VERBOSE:LIB für Diagnose (einmalig einschalten, dann zurück)
option(OTTER_LINK_VERBOSE "Add /VERBOSE:LIB to MSVC link for diagnostics" OFF)
if(MSVC AND OTTER_LINK_VERBOSE)
  target_link_options(mandelbrot_otterdream PRIVATE /VERBOSE:LIB)
endif()

# --- AppLocal: Runtime-DLLs neben die EXE kopieren ----------
add_custom_command(TARGET mandelbrot_otterdream POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_RUNTIME_DLLS:mandelbrot_otterdream>
          $<TARGET_FILE_DIR:mandelbrot_otterdream>
  COMMAND_EXPAND_LISTS
)

# Linux: -pthread explizit
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(mandelbrot_otterdream PRIVATE -pthread)
endif()

# Windows: deterministisch & gehärtet
if(MSVC)
  target_link_options(mandelbrot_otterdream PRIVATE
    $<$<CONFIG:Release>:$<HOST_LINK:/INCREMENTAL:NO /OPT:REF /OPT:ICF>>
    $<$<CONFIG:Release>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
    $<$<CONFIG:RelWithDebInfo>:$<HOST_LINK:/NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf>>
    $<HOST_LINK:/Brepro>
  )
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "HIGHENTROPYVA erfordert 64-bit (/MACHINE:X64). Bitte 64-bit Toolchain verwenden.")
  endif()
endif()

# ---------------------- CUDA 13 Guard ----------------------
if(CUDAToolkit_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "CUDA 13.0 oder neuer ist erforderlich. Gefunden: ${CUDAToolkit_VERSION}")
endif()
find_file(CUDA_GL_INTEROP_HEADER cuda_gl_interop.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
if(NOT CUDA_GL_INTEROP_HEADER)
  message(FATAL_ERROR "cuda_gl_interop.h nicht im CUDAToolkit gefunden.")
endif()

# ---------------------- Defaults ---------------------------
# Default for single-config generators; with multi-config, you still build e.g. RelWithDebInfo explicitly.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
