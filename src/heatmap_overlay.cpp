///// Otter: Kurz & robust: Mini-Heatmap unten rechts, Shader-0 on error, keine verdeckten Pfade.
///// Schneefuchs: Zustands-Restore (VAO/VBO/Program/Blend), gecachte Uniforms, deterministische ASCII-Logs.
/// /// Maus: Keine Marker/Points; y=0 unten; gleiche Datenquelle wie Zoom.
/// /// Datei: src/heatmap_overlay.cpp

#pragma warning(push)
#pragma warning(disable: 4100)

#include "pch.hpp"
#include "heatmap_overlay.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "luchs_log_host.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

#define RS_OFFSET_X(ctx) ((ctx).center.x)
#define RS_OFFSET_Y(ctx) ((ctx).center.y)
#define RS_ZOOM(ctx)     ((ctx).zoom)

namespace HeatmapOverlay {

// --- GL state (TU-lokal) ---
static GLuint sVAO=0, sVBO=0, sProg=0;
static GLint  uScale=-1, uOffset=-1;

// --- Minimaler Shader (Tiles als gefärbte Triangles, vValue in [0,1]) ---
static const char* kVS = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in float aValue;
out float vValue;
uniform vec2 uScale, uOffset;
void main(){
  gl_Position = vec4(aPos * uScale + uOffset, 0.0, 1.0);
  vValue = aValue;
}
)GLSL";

static const char* kFS = R"GLSL(
#version 430 core
in float vValue; out vec4 FragColor;
vec3 map(float v){ float g=smoothstep(0.0,1.0,clamp(v,0.0,1.0));
  return mix(vec3(0.08,0.08,0.10), vec3(1.0,0.6,0.2), g); }
void main(){ FragColor = vec4(map(vValue), 0.85); }
)GLSL";

static GLuint compile(GLenum type, const char* src){
    GLuint sh = glCreateShader(type); if(!sh) return 0;
    glShaderSource(sh,1,&src,nullptr); glCompileShader(sh);
    GLint ok=0; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){ if constexpr(Settings::debugLogging){ GLchar log[512]={0}; glGetShaderInfoLog(sh,512,nullptr,log);
        LUCHS_LOG_HOST("[HM] Shader compile failed: %s", log);} glDeleteShader(sh); return 0; }
    return sh;
}
static GLuint makeProgram(){
    GLuint vs=compile(GL_VERTEX_SHADER,kVS), fs=compile(GL_FRAGMENT_SHADER,kFS);
    if(!vs||!fs){ if(vs)glDeleteShader(vs); if(fs)glDeleteShader(fs); return 0; }
    GLuint p=glCreateProgram(); if(!p){ glDeleteShader(vs); glDeleteShader(fs); return 0; }
    glAttachShader(p,vs); glAttachShader(p,fs); glLinkProgram(p);
    glDeleteShader(vs); glDeleteShader(fs);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ if constexpr(Settings::debugLogging){ GLchar log[512]={0}; glGetProgramInfoLog(p,512,nullptr,log);
        LUCHS_LOG_HOST("[HM] Program link failed: %s", log);} glDeleteProgram(p); return 0; }
    return p;
}

// --- API ---
void toggle(RendererState& ctx){
#if defined(USE_HEATMAP_OVERLAY)
    ctx.heatmapOverlayEnabled = !ctx.heatmapOverlayEnabled;
#else
    (void)ctx;
#endif
}

void cleanup(){
    if(sVAO) glDeleteVertexArrays(1,&sVAO);
    if(sVBO) glDeleteBuffers(1,&sVBO);
    if(sProg) glDeleteProgram(sProg);
    sVAO=sVBO=sProg=0; uScale=uOffset=-1;
}

void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height, int tileSize,
                 [[maybe_unused]] GLuint textureId,
                 RendererState& ctx)
{
    if(!ctx.heatmapOverlayEnabled) return;
    if(width<=0||height<=0||tileSize<=0) return;

    const int tilesX = (width  + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int nTiles = tilesX * tilesY;
    if((int)entropy.size()<nTiles || (int)contrast.size()<nTiles) return;

    // Lazy-Init
    if(!sProg){
        sProg = makeProgram();
        if(!sProg){ if constexpr(Settings::debugLogging) LUCHS_LOG_HOST("[HM] ERROR: program==0"); return; }
        uScale  = glGetUniformLocation(sProg,"uScale");
        uOffset = glGetUniformLocation(sProg,"uOffset");
    }
    if(!sVAO){ glGenVertexArrays(1,&sVAO); glGenBuffers(1,&sVBO); }

    // Wertebereich bestimmen
    float maxVal = 1e-6f;
    for(int i=0;i<nTiles;++i) maxVal = std::max(maxVal, entropy[i]+contrast[i]);

    // Triangles bauen: pro Tile 6 Vertices, (x,y,v) mit y=0 unten
    std::vector<float> verts;
    verts.reserve(size_t(nTiles)*6u*3u);
    for(int y=0;y<tilesY;++y){
        for(int x=0;x<tilesX;++x){
            const int i = y*tilesX + x;
            const float v = (entropy[i]+contrast[i]) / maxVal;
            const float px=float(x), py=float(y);
            const float q[6][3] = {
                {px,py,v},{px+1,py,v},{px+1,py+1,v},
                {px,py,v},{px+1,py+1,v},{px,py+1,v}
            };
            verts.insert(verts.end(), &q[0][0], &q[0][0]+18);
        }
    }

    // Platzierung: unten rechts, Höhe ~100px, Seitenverhältnis = tilesX/tilesY
    constexpr int hPx=100, pad=16;
    const float aspect = tilesY>0 ? float(tilesX)/float(tilesY) : 1.0f;
    const int   wPx = std::max(1, int(std::round(hPx*aspect)));

    const float sx = (float(wPx)/float(width)/float(tilesX))*2.0f;
    const float sy = (float(hPx)/float(height)/float(tilesY))*2.0f;
    const float ox =  1.0f - (float(wPx + pad)/float(width))*2.0f;
    const float oy = -1.0f + (float(pad)      /float(height))*2.0f;

    // State sichern
    GLint prevVAO=0, prevBuf=0, prevProg=0, srcRGB=0,dstRGB=0,srcA=0,dstA=0;
    GLboolean wasBlend=GL_FALSE;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING,&prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&prevBuf);
    glGetIntegerv(GL_CURRENT_PROGRAM,&prevProg);
    glGetBooleanv(GL_BLEND,&wasBlend);
    glGetIntegerv(GL_BLEND_SRC_RGB,&srcRGB); glGetIntegerv(GL_BLEND_DST_RGB,&dstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA,&srcA); glGetIntegerv(GL_BLEND_DST_ALPHA,&dstA);

    // Zeichnen
    glUseProgram(sProg);
    glUniform2f(uScale,sx,sy);
    glUniform2f(uOffset,ox,oy);

    glBindVertexArray(sVAO);
    glBindBuffer(GL_ARRAY_BUFFER,sVBO);
    glBufferData(GL_ARRAY_BUFFER,GLsizeiptr(verts.size()*sizeof(float)),nullptr,GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER,0,GLsizeiptr(verts.size()*sizeof(float)),verts.data());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)(2*sizeof(float)));

    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES,0,GLsizei(verts.size()/3));

    // Restore
    glBlendFuncSeparate(srcRGB,dstRGB,srcA,dstA);
    if(!wasBlend) glDisable(GL_BLEND);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER,prevBuf);
    glBindVertexArray((GLuint)prevVAO);
    glUseProgram((GLuint)prevProg);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[HM] draw ok tiles=%dx%d verts=%zu zoom=%.6f center=(%.9f,%.9f)",
                       tilesX,tilesY,verts.size()/3, RS_ZOOM(ctx), RS_OFFSET_X(ctx), RS_OFFSET_Y(ctx));
    }
}

} // namespace HeatmapOverlay

#pragma warning(pop)
