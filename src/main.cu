// File: src/main.cu
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
#include <cstdlib>

// CUDA-Error-Check
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Complexity-Kernel (unverändert)
__global__ void computeComplexity(const uchar4* img,
                                  int width, int height,
                                  float* complexity);

// Simple Vertex/Fragment Shader
static const char* vertexShaderSrc = R"glsl(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;
out vec2 vTex;
void main(){
  gl_Position = vec4(aPos,0,1);
  vTex = aTex;
}
)glsl";
static const char* fragmentShaderSrc = R"glsl(
#version 330 core
in vec2 vTex;
out vec4 fColor;
uniform sampler2D uTex;
void main(){
  fColor = texture(uTex, vTex);
}
)glsl";

int main(){
  const int width  = 1024;
  const int height = 768;
  size_t imgBytes  = width*height*sizeof(uchar4);

  float zoom    = 300.f;
  float2 offset = make_float2(0,0);
  int maxIter   = 500;

  // GLFW + GL Context
  if(!glfwInit()) return EXIT_FAILURE;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
  glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* w = glfwCreateWindow(width,height,"Auto-Zoom",nullptr,nullptr);
  if(!w) return EXIT_FAILURE;
  glfwMakeContextCurrent(w);

  if(glewInit()!=GLEW_OK){
    std::cerr<<"GLEW failed\n"; return EXIT_FAILURE;
  }

  // *** Neu: Viewport & Pixel-Store ***
  glViewport(0,0,width,height);
  glPixelStorei(GL_UNPACK_ALIGNMENT,1);

  // PBO + CUDA Interop
  GLuint pbo;
  glGenBuffers(1,&pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER,imgBytes,nullptr,GL_DYNAMIC_DRAW);

  cudaGraphicsResource* cudaPbo = nullptr;
  checkCuda(cudaGraphicsGLRegisterBuffer(&cudaPbo,pbo,cudaGraphicsMapFlagsWriteDiscard),
            "cudaGraphicsGLRegisterBuffer");

  // Texture
  GLuint tex;
  glGenTextures(1,&tex);
  glBindTexture(GL_TEXTURE_2D,tex);
  glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);

  // Shader-Compilation
  auto comp = [&](GLenum t,const char* src){
    GLuint s=glCreateShader(t);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){
      char buf[512]; glGetShaderInfoLog(s,512,nullptr,buf);
      std::cerr<<buf; std::exit(EXIT_FAILURE);
    }
    return s;
  };
  GLuint vs=comp(GL_VERTEX_SHADER,vertexShaderSrc);
  GLuint fs=comp(GL_FRAGMENT_SHADER,fragmentShaderSrc);
  GLuint prog=glCreateProgram();
  glAttachShader(prog,vs);
  glAttachShader(prog,fs);
  glLinkProgram(prog);
  { GLint ok; glGetProgramiv(prog,GL_LINK_STATUS,&ok);
    if(!ok){
      char buf[512]; glGetProgramInfoLog(prog,512,nullptr,buf);
      std::cerr<<buf; std::exit(EXIT_FAILURE);
    }
  }
  glDeleteShader(vs); glDeleteShader(fs);

  // Quad VAO/VBO
  float quad[] = {
    -1,-1, 0,0,
     1,-1, 1,0,
    -1, 1, 0,1,
     1, 1, 1,1,
  };
  GLuint VAO, VBO;
  glGenVertexArrays(1,&VAO);
  glGenBuffers(1,&VBO);
  glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER,VBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(quad),quad,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
  glBindVertexArray(0);

  // Complexity-Buffer
  int tilesX=(width+TILE_W-1)/TILE_W;
  int tilesY=(height+TILE_H-1)/TILE_H;
  int nTiles=tilesX*tilesY;
  float* d_comp=nullptr;
  checkCuda(cudaMalloc(&d_comp,nTiles*sizeof(float)),"cudaMalloc comp");
  std::vector<float> h_comp(nTiles);

  // Mainloop
  while(!glfwWindowShouldClose(w)){
    // 1) reset comp
    checkCuda(cudaMemset(d_comp,0,nTiles*sizeof(float)),"memset");

    // 2) map PBO
    uchar4* d_img=nullptr; size_t s=0;
    checkCuda(cudaGraphicsMapResources(1,&cudaPbo,0),"MapRes");
    checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_img,&s,cudaPbo),
              "GetPtr");

    // 3) Mandelbrot
    dim3 bd(TILE_W,TILE_H), gd(tilesX,tilesY);
    mandelbrotHybrid<<<gd,bd>>>(d_img,width,height,zoom,offset,maxIter);
    checkCuda(cudaDeviceSynchronize(),"mandelbrotHybrid");

    // 4) complexity
    computeComplexity<<<gd,bd>>>(d_img,width,height,d_comp);
    checkCuda(cudaDeviceSynchronize(),"computeComplexity");

    // 5) unmap PBO
    checkCuda(cudaGraphicsUnmapResources(1,&cudaPbo,0),"Unmap");

    // 6) *** NEU: PBO → Textur ***
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo);
    glBindTexture(GL_TEXTURE_2D,tex);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);

    // 7) find best tile & zoom…
    checkCuda(cudaMemcpy(h_comp.data(),d_comp,nTiles*sizeof(float),cudaMemcpyDeviceToHost),
              "cpy comp");
    int bi=0; float bs=-1.f;
    for(int i=0;i<nTiles;++i){
      if(h_comp[i]>bs){ bs=h_comp[i]; bi=i;}
    }
    int bx=bi%tilesX, by=bi/tilesX;
    offset.x += ((bx+0.5f)*TILE_W - width*0.5f)/zoom;
    offset.y += ((by+0.5f)*TILE_H - height*0.5f)/zoom;
    zoom *= 1.2f;

    // 8) draw quad
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);
    glBindVertexArray(VAO);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D,tex);
      glUniform1i(glGetUniformLocation(prog,"uTex"),0);
      glDrawArrays(GL_TRIANGLE_STRIP,0,4);
    glBindVertexArray(0);

    glfwSwapBuffers(w);
    glfwPollEvents();
  }

  // cleanup...
  cudaFree(d_comp);
  cudaGraphicsUnregisterResource(cudaPbo);
  glDeleteBuffers(1,&pbo);
  glDeleteTextures(1,&tex);
  glDeleteProgram(prog);
  glDeleteBuffers(1,&VBO);
  glDeleteVertexArrays(1,&VAO);
  glfwDestroyWindow(w);
  glfwTerminate();
  return 0;
}
