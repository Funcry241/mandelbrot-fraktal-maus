cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# ---- Early: nur Release/Debug und Imported-Config-Mapping ----
if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "Supported configs" FORCE)
endif()
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL     Release)
set(CMAKE_MAP_IMPORTED_CONFIG_RelWithDebInfo Release)
set(CMAKE_MAP_IMPORTED_CONFIG_MinSizeRel     Release)
set(CMAKE_MAP_IMPORTED_CONFIG_PROFILE        Release)
set(CMAKE_MAP_IMPORTED_CONFIG_CHECKED        Release)
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)

project(mandelbrot_otterdream LANGUAGES CXX CUDA)

# MSVC Runtime über MSVC_RUNTIME_LIBRARY steuerbar machen
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------- Ninja QoL --------------------------
if(CMAKE_GENERATOR MATCHES "Ninja")
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON)
  set_property(GLOBAL PROPERTY JOB_POOLS console_pool=1 link_pool=1)
  set(CMAKE_JOB_POOL_LINK link_pool)
endif()

# ---------------------- Archs & Standards ------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90 CACHE STRING "Default architectures if none provided")
endif()

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

# ---------------------- Pakete ------------------------------
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Windows: direkt opengl32, sonst OpenGL::GL
if(WIN32)
  set(_OPENGL_GL_LIB opengl32)
else()
  set(_OPENGL_GL_LIB OpenGL::GL)
endif()

# --- GLFW --------------------------------------------------
set(GLFW_LINK_ITEM "")
find_package(glfw3 CONFIG REQUIRED)
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

# --- GLEW (dynamic, no GLU) --------------------------------
set(GLEW_USE_STATIC_LIBS OFF CACHE BOOL "Link GLEW dynamically" FORCE)
remove_definitions(-DGLEW_STATIC)

set(GLEW_LINK_ITEM "")
find_package(GLEW CONFIG REQUIRED)
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
    REQUIRED)
endif()

# -------- Helper: GLU aus importierten Targets entfernen ----
function(otter_strip_glu_from target_name)
  if(NOT TARGET ${target_name})
    return()
  endif()
  # Prüfe und säubere gängige Link-Properties
  foreach(_prop
          INTERFACE_LINK_LIBRARIES
          INTERFACE_LINK_LIBRARIES_DEBUG
          INTERFACE_LINK_LIBRARIES_RELEASE
          IMPORTED_LINK_INTERFACE_LIBRARIES
          IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG
          IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE)
    get_target_property(_libs ${target_name} ${_prop})
    if(_libs)
      set(_new "${_libs}")
      list(FILTER _new EXCLUDE REGEX [[(^|;)OpenGL::GLU($|;)]])
      list(FILTER _new EXCLUDE REGEX [[(^|;)[Gg][Ll][Uu]32(\.lib)?($|;)]])
      if(NOT "${_new}" STREQUAL "${_libs}")
        set_property(TARGET ${target_name} PROPERTY ${_prop} "${_new}")
      endif()
    endif()
  endforeach()
endfunction()

# konsequent auf die üblichen Verdächtigen anwenden
otter_strip_glu_from(${GLEW_LINK_ITEM})
otter_strip_glu_from(${GLFW_LINK_ITEM})
if(TARGET OpenGL::GLU)
  otter_strip_glu_from(OpenGL::GLU)
endif()

# ---------------------- Features ---------------------------
option(OTTER_USE_ORT   "Enable ONNX Runtime AOP (Policy-Replikatoren)" OFF)
option(OTTER_USE_NVRTC "Enable NVRTC Coloring (Color-Replikatoren)"   OFF)
option(OTTER_LINK_CUDA_DRIVER "Link the CUDA Driver API (cu*). Enable only if you call cu* symbols." OFF)

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
  # Axolotel HUD
  src/axolotel_hud.cpp
  src/axolotel_coupler.cpp
  # Dachs HUD
  src/dachs_hud.cpp
  # Replikatoren (core)
  src/perturb_core.cpp
  src/perturb_orbit_mp.cpp
  src/luchs_dsl.cpp
)

# Linker-Sprache sichtbar und korrekt
set_property(TARGET mandelbrot_otterdream PROPERTY LINKER_LANGUAGE CXX)

# Optionale Replikatoren-Quellen
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
  set_source_files_properties(src/perturb_core.cpp PROPERTIES COMPILE_OPTIONS "/wd4702")
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

# ---------------------- Includes & Defines -----------------
target_include_directories(mandelbrot_otterdream PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(mandelbrot_otterdream PRIVATE
  GLEW_NO_IMAGING
  GLEW_NO_GLU
  CAPY_DEFAULT_ON=1
  $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX WIN32_LEAN_AND_MEAN>
  $<$<COMPILE_LANGUAGE:CUDA>:__NV_NO_VECTOR_DEPRECATION_DIAG>
  $<$<BOOL:${OTTER_USE_ORT}>:OTTER_USE_ORT=1>
  $<$<BOOL:${OTTER_USE_NVRTC}>:OTTER_USE_NVRTC=1>
)

# ---------------------- Link (Basis) -----------------------
target_link_libraries(mandelbrot_otterdream PRIVATE
  ${_OPENGL_GL_LIB}
  ${GLEW_LINK_ITEM}
  ${GLFW_LINK_ITEM}
  Threads::Threads
)

if(WIN32)
  target_link_libraries(mandelbrot_otterdream PRIVATE user32 gdi32 shell32 advapi32)
endif()

# --- CUDA Driver API (optional) ----------------------------
if(OTTER_LINK_CUDA_DRIVER)
  set(_USE_CUDA_DRIVER_TARGET OFF)
  if(TARGET CUDA::cuda_driver)
    foreach(_cfg IN ITEMS RELEASE DEBUG)
      get_target_property(_drv_loc CUDA::cuda_driver IMPORTED_LOCATION_${_cfg})
      if(_drv_loc)
        set(_USE_CUDA_DRIVER_TARGET ON)
      endif()
    endforeach()
  endif()
  if(_USE_CUDA_DRIVER_TARGET)
    target_link_libraries(mandelbrot_otterdream PRIVATE CUDA::cuda_driver)
  else()
    find_library(CUDA_DRIVER_LIB
      NAMES cuda nvcuda
      HINTS
        "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_LIBRARY_DIR}/x64"
        "$ENV{CUDA_PATH}/lib/x64"
        "$ENV{CUDA_PATH_V13_2}/lib/x64" "$ENV{CUDA_PATH_V13_1}/lib/x64" "$ENV{CUDA_PATH_V13_0}/lib/x64"
        "$ENV{SystemRoot}/System32"
    )
    if(CUDA_DRIVER_LIB)
      target_link_libraries(mandelbrot_otterdream PRIVATE "${CUDA_DRIVER_LIB}")
    else()
      message(STATUS "CUDA Driver API nicht gefunden – überspringe Verlinkung (cu* ungenutzt).")
    endif()
  endif()
endif()

# --- Hardcore Anti-GLU (explizit + transitiv) --------------
if(WIN32)
  get_target_property(_OTTER_LINK_LIBS mandelbrot_otterdream LINK_LIBRARIES)
  if(_OTTER_LINK_LIBS)
    set(_OTTER_LINK_LIBS_FILTERED "${_OTTER_LINK_LIBS}")
    list(FILTER _OTTER_LINK_LIBS_FILTERED EXCLUDE REGEX [[(^|;)OpenGL::GLU($|;)]])
    list(FILTER _OTTER_LINK_LIBS_FILTERED EXCLUDE REGEX [[(^|;)[Gg][Ll][Uu]32(\.lib)?($|;)]])
    if(NOT _OTTER_LINK_LIBS_FILTERED STREQUAL "${_OTTER_LINK_LIBS}")
      set_property(TARGET mandelbrot_otterdream PROPERTY LINK_LIBRARIES "${_OTTER_LINK_LIBS_FILTERED}")
    endif()
  endif()
  if(MSVC)
    # Falls irgendein DEFAULTLIB Eintrag GLU reinmogelt:
    target_link_options(mandelbrot_otterdream PRIVATE /NODEFAULTLIB:GLU32 /NODEFAULTLIB:glu32)
  endif()
endif()

# ---------------------- NVCC Host-Compiler PIN --------------
# Stabiler -ccbin, auch wenn PATH/Runner-Kaskade variiert.
if(MSVC)
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Host compiler for NVCC" FORCE)
  endif()
  # Doppelt hält besser: in Flags aufnehmen, damit der dlink es garantiert sieht.
  if(CMAKE_CXX_COMPILER)
    # Sauber quoten (Spaces in Pfaden)
    string(REPLACE "/" "\\" _ccbin_norm "${CMAKE_CXX_COMPILER}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=\"${_ccbin_norm}\"" CACHE STRING "NVCC flags" FORCE)
  endif()
endif()

# ---------------------- Link-Optionen ----------------------
option(OTTER_LINK_VERBOSE "Add /VERBOSE:LIB to MSVC link for diagnostics" ON)
if(MSVC)
  if(OTTER_LINK_VERBOSE)
    target_link_options(mandelbrot_otterdream PRIVATE /VERBOSE:LIB)
  endif()
  target_link_options(mandelbrot_otterdream PRIVATE
    /NXCOMPAT /DYNAMICBASE /HIGHENTROPYVA /guard:cf /Brepro
    $<$<CONFIG:Release>:/INCREMENTAL:NO>
    $<$<CONFIG:Release>:/OPT:REF>
    $<$<CONFIG:Release>:/OPT:ICF>
  )
  if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "HIGHENTROPYVA erfordert 64-bit (/MACHINE:X64). Bitte 64-bit Toolchain verwenden.")
  endif()
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

# Diagnose
message(STATUS "Otter Configs: ${CMAKE_CONFIGURATION_TYPES}")
message(STATUS "Map RelWithDebInfo -> ${CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO}")
get_target_property(_OTTER_LINK_LANG mandelbrot_otterdream LINKER_LANGUAGE)
message(STATUS "Otter Linker language: ${_OTTER_LINK_LANG}")
