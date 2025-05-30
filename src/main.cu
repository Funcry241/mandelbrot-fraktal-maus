// Datei: src/main.cu
// Maus-Kommentar: Host-Code erstellt Bildpuffer, startet CUDA-Kernel, zeigt das Ergebnis
// mit modernem OpenGL (Shader + VBO/VAO) an und übernimmt adaptives Zoom-Feedback.

#include "core_kernel.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <iostream>
#include <vector>
#include <limits>
#include <string>

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkShader(GLuint shader) {
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        char buf[512];
        glGetShaderInfoLog(shader, 512, nullptr, buf);
        std::cerr << "Shader-Fehler: " << buf << std::endl;
        exit(EXIT_FAILURE);
    }
}
void checkProgram(GLuint prog) {
    GLint status;
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        char buf[512];
        glGetProgramInfoLog(prog, 512, nullptr, buf);
        std::cerr << "Program-Link-Fehler: " << buf << std::endl;
        exit(EXIT_FAILURE);
    }
}

const char* vertSrc = R"glsl(
#version 330 core
layout(location = 0) in vec2 pos;
out vec2 uv;
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)glsl";

const char* fragSrc = R"glsl(
#version 330 core
in vec2 uv;
out vec4 outColor;
uniform sampler2D tex;
void main() {
    outColor = texture(tex, uv);
}
)glsl";

int main() {
    std::cout << "=== Programm gestartet ===\n";
    const int W = 1024, H = 768;
    size_t imgBytes = W * H * sizeof(uchar4);
    float zoom = 300.0f;
    float2 offset = make_float2(0.0f, 0.0f);
    int maxIter = 500;

    if (!glfwInit()) exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(W, H, "Auto-Zoom Mandelbrot", nullptr, nullptr);
    glfwMakeContextCurrent(win);
    if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed\n"; return EXIT_FAILURE; }
    std::cout << "Setup abgeschlossen, betrete Haupt-Loop\n";

    // Shader
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertSrc, nullptr);
    glCompileShader(vs); checkShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragSrc, nullptr);
    glCompileShader(fs); checkShader(fs);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs); glAttachShader(prog, fs);
    glLinkProgram(prog); checkProgram(prog);
    glDeleteShader(vs); glDeleteShader(fs);

    // Fullscreen-Quad
    float quadVerts[] = {
        -1.0f,-1.0f,  1.0f,-1.0f,  1.0f, 1.0f,
        -1.0f,-1.0f,  1.0f, 1.0f, -1.0f, 1.0f
    };
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);

    // PBO + CUDA
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imgBytes, nullptr, GL_DYNAMIC_DRAW);
    cudaGraphicsResource* cudaPbo;
    checkCuda(cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsMapFlagsWriteDiscard), "cudaGraphicsGLRegisterBuffer");

    // Texture
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,W,H,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);

    int tilesX = (W + TILE_W - 1)/TILE_W;
    int tilesY = (H + TILE_H - 1)/TILE_H;
    int total = tilesX*tilesY;
    float* d_comp;
    checkCuda(cudaMalloc(&d_comp, total*sizeof(float)), "cudaMalloc comp");
    std::vector<float> h_comp(total);

    int frame=0;
    while(!glfwWindowShouldClose(win)){
        std::cout<<"Frame "<<frame++<<": Zoom="<<zoom<<" Offset=("<<offset.x<<","<<offset.y<<")\n";
        checkCuda(cudaMemset(d_comp,0,total*sizeof(float)),"memset comp");

        uchar4* d_img;
        size_t sz;
        checkCuda(cudaGraphicsMapResources(1,&cudaPbo,0),"MapRes");
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_img,&sz,cudaPbo),"GetPtr");

        dim3 bd(TILE_W,TILE_H), gd(tilesX,tilesY);
        launch_mandelbrotHybrid(d_img,W,H,zoom,offset,maxIter);
        checkCuda(cudaGetLastError(),"mandelbrotHybrid");

        computeComplexity<<<gd,bd>>>(d_img,W,H,d_comp);
        checkCuda(cudaGetLastError(),"computeComplexity");
        checkCuda(cudaDeviceSynchronize(),"synchronize");

        checkCuda(cudaGraphicsUnmapResources(1,&cudaPbo,0),"Unmap");
        checkCuda(cudaMemcpy(h_comp.data(),d_comp,total*sizeof(float),cudaMemcpyDeviceToHost),"Memcpy comp");

        int bestIdx=0; float best= -1;
        for(int i=0;i<total;++i) if(h_comp[i]>best){best=h_comp[i];bestIdx=i;}
        std::cout<<"  bestScore="<<best<<"\n";
        int bx=bestIdx%tilesX, by=bestIdx/tilesX;
        offset.x+=((bx+0.5f)*TILE_W - W*0.5f)/zoom;
        offset.y+=((by+0.5f)*TILE_H - H*0.5f)/zoom;
        zoom*=1.2f;

        // Draw
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo);
        glBindTexture(GL_TEXTURE_2D,tex);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,W,H,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);

        glUseProgram(prog);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES,0,6);
        glBindVertexArray(0);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    cudaFree(d_comp);
    cudaGraphicsUnregisterResource(cudaPbo);
    glDeleteBuffers(1,&pbo);
    glDeleteTextures(1,&tex);
    glDeleteBuffers(1,&VBO);
    glDeleteVertexArrays(1,&VAO);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
