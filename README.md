# OtterDream Mandelbrot Renderer

🐭 CUDA + OpenGL Renderer für das Mandelbrot-Fraktal mit intelligentem Auto-Zoom.

---

## Features
- 🚀 **CUDA-Accelerated**: Echtzeit-Rendering mit GPU-Boost.
- 🧠 **Auto-Zoom**: Fokussiert auf Bereiche mit maximaler Variabilität.
- 🎨 **High-Quality Coloring**: Log-basiertes Smoothing für sanfte Farbübergänge.
- 🖥️ **OpenGL-Interop**: Effiziente Framebuffer-Updates über PBOs.
- 🛠️ **Debug Mode**: Optionale Gradienten-Visualisierung.

---

## Build Instructions

1. **Clone repository**:
   ```bash
   git clone https://github.com/dein-username/otterdream-mandelbrot.git
   cd otterdream-mandelbrot
   ```

2. **Setup vcpkg**:
   ```bash
   git clone https://github.com/microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.bat
   vcpkg integrate install
   vcpkg install glfw3 glew
   ```

3. **Build**:
   ```bash
   mkdir build
   cd build
   cmake .. -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
   cmake --build .
   ```

4. **Run**:
   ```bash
   ./OtterDreamMandelbrot
   ```

---

## Controls
| Key | Action         |
|:---:|----------------|
| `Arrow Keys` | Pan View     |
| `+ / -`     | Zoom In/Out  |
| `ESC`       | Exit Program |

---

## Dependencies
- CUDA 12+
- OpenGL 4.3+
- GLEW
- GLFW
- vcpkg (for package management)

---

## License
MIT License — free for personal and commercial use.

---

## Screenshot
![Mandelbrot Screenshot](assets/mandelbrot_example.png)
