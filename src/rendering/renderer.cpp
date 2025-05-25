// src/rendering/renderer.cpp – korrigiert: hi/lo korrekt berechnet ohne dd_real

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <cmath>

#include "rendering/renderer.hpp"
#include "settings.hpp"
#include "metrics.hpp"
#include "utils/cuda_utils.hpp"
#include "mandelbrot.hpp"
#include "gui.hpp"

extern Settings S;
extern Metrics   M;

static const char* quadVertexSrc = R"( 
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;
out vec2 vTex;
void main() {
    vTex = aTex;
    gl_Position = vec4(aPos,0.0,1.0);
}
)";
static const char* quadFragmentSrc = R"(
#version 330 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vTex);
}
)";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader,1,&src,nullptr);
    glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512];
        glGetShaderInfoLog(shader, 512, nullptr, buf);
        std::cerr<<"Shader-Error: "<<buf<<"\n";
        std::exit(EXIT_FAILURE);
    }
    return shader;
}

Renderer::Renderer(GLFWwindow* win)
 : window(win)
{
    initGL();
    setupShaders();
    setupBuffers();
}

Renderer::~Renderer() {
    // Optional: Ressourcen freigeben
}

void Renderer::initGL() {
    glewInit();
    glClearColor(0.0f,0.0f,0.0f,1.0f);
    glViewport(0,0,S.width,S.height);
    glDisable(GL_DEPTH_TEST);
}

void Renderer::setupShaders() {
    GLuint vs = compileShader(GL_VERTEX_SHADER,   quadVertexSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, quadFragmentSrc);
    quadProg = glCreateProgram();
    glAttachShader(quadProg, vs);
    glAttachShader(quadProg, fs);
    glLinkProgram(quadProg);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

void Renderer::setupBuffers() {
    glGenBuffers(1,&pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,S.width*S.height*sizeof(uchar4), nullptr,GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);

    checkCuda(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource,pbo,cudaGraphicsMapFlagsWriteDiscard),"register PBO");

    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D,tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,S.width,S.height,0,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);

    float verts[] = {
        -1,-1, 0,0,
         1,-1, 1,0,
        -1, 1, 0,1,
         1, 1, 1,1
    };
    glGenVertexArrays(1,&quadVAO);
    glGenBuffers(1,&quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER,quadVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glBindVertexArray(0);
}

void Renderer::pollEvents() {
    glfwPollEvents();
}

void Renderer::renderFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    uchar4* devPtr=nullptr; size_t sz=0;
    checkCuda(cudaGraphicsMapResources(1,&cuda_pbo_resource,0),"map");
    checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&devPtr,&sz,cuda_pbo_resource),"getptr");

    double dx_hi = S.offsetX.convert_to<double>();
    double dx_lo = (S.offsetX - dx_hi).convert_to<double>();
    double dy_hi = S.offsetY.convert_to<double>();
    double dy_lo = (S.offsetY - dy_hi).convert_to<double>();

    launch_kernel_dd(devPtr, S.width, S.height,
                     S.zoom.convert_to<double>(),
                     dx_hi, dx_lo,
                     dy_hi, dy_lo,
                     S.maxIter);

    checkCuda(cudaGraphicsUnmapResources(1,&cuda_pbo_resource,0),"unmap");

    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo);
    glBindTexture(GL_TEXTURE_2D,tex);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,S.width,S.height,GL_RGBA,GL_UNSIGNED_BYTE,nullptr);
    glUseProgram(quadProg);
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP,0,4);

    render_gui(S,M);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
}

const Settings& Renderer::getSettings() const {
    return S;
}
