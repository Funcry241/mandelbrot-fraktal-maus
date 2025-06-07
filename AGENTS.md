# OtterDream Mandelbrot – AGENTS.md

## Overview

This project is a CUDA-accelerated Mandelbrot fractal renderer using OpenGL for display.

It supports:
- Automatic zoom into visually complex areas.
- Real-time rendering with GPU acceleration (CUDA + OpenGL).
- Smooth progressive refinement of fractal details.
- Optional debug logging and frame metrics.

> *Note: HUD overlay planned for future versions.*

---

## Build Agents

| Agent               | Purpose                           | Trigger             | Actions                       |
|---------------------|-----------------------------------|---------------------|-------------------------------|
| GitHub Copilot       | Code suggestions                 | In-Editor           | Supports C++, CUDA, CMake     |
| Dependabot          | Dependency management             | Weekly              | Monitors `vcpkg.json`         |
| GitHub Actions (CI)  | Build validation (planned)        | Push, PR to `main`  | CMake configure, Ninja build  |

---

## Tools and Versions

| Tool            | Minimum Version | Notes                      |
|-----------------|-----------------|----------------------------|
| CUDA Toolkit    | 12.0+            | Required for GPU rendering |
| OpenGL          | 4.3+             | Required for shaders       |
| CMake           | 3.24+            | Modern CMake build         |
| vcpkg           | Manifest mode    | Dependency management      |
| Ninja           | Latest           | Fast parallel builds       |

---

## GPU Compatibility

- Requires Compute Capability **≥ 3.0**.
- No need for Dynamic Parallelism.
- Recommended: NVIDIA RTX 20xx or newer.

---

## Local Build Instructions

```bash
git clone <repo-url>
cd mandelbrot_otterdream
vcpkg install
.\build.ps1
