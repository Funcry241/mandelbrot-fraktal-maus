///// Otter: Projekt „Pfau“ – kompakt: vereinheitlichtes Panel (Alpha/Margin/Radius) + Tiles (TR).
///// Schneefuchs: Keine API-Änderung; State-Restore wie zuvor; ASCII-Logs mit [UI/Pfau].
///// Maus: Tiles opak, Panel halbtransparent; Pixel-Snapping; identische Top-Margin wie WS.
// ///// Datei: src/heatmap_overlay.cpp
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

static GLuint sVAO=0, sVBO=0, sProg=0;                 // Tiles
static GLint  uScale=-1, uOffset=-1;
static GLuint sPanelVAO=0, sPanelVBO=0, sPanelProg=0;  // Panel
static GLint  uViewportPx=-1, uPanelRectPx=-1, uRadiusPx=-1, uAlpha=-1, uBorderPx=-1;

static const char* kVS = R"GLSL(#version 430 core
layout(location=0) in vec2 aPos; layout(location=1) in float aValue;
out float vValue; uniform vec2 uScale,uOffset;
void main(){ gl_Position=vec4(aPos*uScale+uOffset,0,1); vValue=aValue; }
)GLSL";

static const char* kFS = R"GLSL(#version 430 core
in float vValue; out vec4 FragColor;
vec3 map(float v){ float g=smoothstep(0,1,clamp(v,0,1));
  return mix(vec3(0.08,0.08,0.10), vec3(1.0,0.82,0.32), g); }
void main(){ FragColor=vec4(map(vValue),1.0); } // Tiles opak; Panel liefert Alpha
)GLSL";

static const char* kPanelVS = R"GLSL(#version 430 core
layout(location=0) in vec2 aPosPx; layout(location=1) in vec3 aColor;
uniform vec2 uViewportPx; out vec3 vColor; out vec2 vPx;
vec2 toNdc(vec2 p, vec2 vp){ return vec2(p.x/vp.x*2-1, 1-p.y/vp.y*2); }
void main(){ vColor=aColor; vPx=aPosPx; gl_Position=vec4(toNdc(aPosPx,uViewportPx),0,1); }
)GLSL";

static const char* kPanelFS = R"GLSL(#version 430 core
in vec3 vColor; in vec2 vPx; out vec4 FragColor;
uniform vec4 uPanelRectPx; uniform float uRadiusPx,uAlpha,uBorderPx;

float sdRoundRect(vec2 p, vec2 b, float r){
  vec2 d = abs(p) - b + vec2(r);
  return length(max(d,0.0)) - r;      // d<0 innen, d>0 außen
}

void main(){
  vec2 c = 0.5 * (uPanelRectPx.xy + uPanelRectPx.zw);
  vec2 b = 0.5 * (uPanelRectPx.zw - uPanelRectPx.xy);
  float d = sdRoundRect(vPx - c, b, uRadiusPx);

  // One-sided AA: innen stets voll deckend, nur außen ausfaden
  float aa   = fwidth(d);
  float body = 1.0 - smoothstep(0.0, aa, max(d, 0.0));  // 1 innen, 0 außen

  // Sehr dezenter innerer Stroke direkt an der Kante (kein Halo im Padding)
  float inner = smoothstep(-uBorderPx, 0.0, d);         // 0 tief innen → 1 an Kante
  vec3  borderCol = vec3(1.0, 0.82, 0.32);
  vec3  col = mix(vColor, borderCol, 0.12 * inner);     // 12% statt 20%

  FragColor = vec4(col, uAlpha * body);
}
)GLSL";

static GLuint make(GLenum type, const char* src){
    GLuint sh=glCreateShader(type); if(!sh) return 0;
    glShaderSource(sh,1,&src,nullptr); glCompileShader(sh);
    GLint ok=0; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){ if constexpr(Settings::debugLogging){ char log[512]{}; glGetShaderInfoLog(sh,512,nullptr,log);
        LUCHS_LOG_HOST("[UI/Pfau][HM] shader compile fail: %s", log);} glDeleteShader(sh); return 0; }
    return sh;
}
static GLuint program(const char* vs, const char* fs){
    GLuint v=make(GL_VERTEX_SHADER,vs), f=make(GL_FRAGMENT_SHADER,fs);
    if(!v||!f){ if(v)glDeleteShader(v); if(f)glDeleteShader(f); return 0; }
    GLuint p=glCreateProgram(); if(!p){ glDeleteShader(v); glDeleteShader(f); return 0; }
    glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ if constexpr(Settings::debugLogging){ char log[512]{}; glGetProgramInfoLog(p,512,nullptr,log);
        LUCHS_LOG_HOST("[UI/Pfau][HM] program link fail: %s", log);} glDeleteProgram(p); return 0; }
    return p;
}

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

    if(sPanelVAO) glDeleteVertexArrays(1,&sPanelVAO);
    if(sPanelVBO) glDeleteBuffers(1,&sPanelVBO);
    if(sPanelProg) glDeleteProgram(sPanelProg);
    sPanelVAO=sPanelVBO=sPanelProg=0;
    uViewportPx=uPanelRectPx=uRadiusPx=uAlpha=uBorderPx=-1;
}

void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height, int tileSize,
                 [[maybe_unused]] GLuint textureId,
                 RendererState& ctx)
{
    if(!ctx.heatmapOverlayEnabled || width<=0||height<=0||tileSize<=0) return;

    const int tilesX = (width  + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int nTiles = tilesX * tilesY;
    if((int)entropy.size()<nTiles || (int)contrast.size()<nTiles) return;

    if(!sProg){
        sProg = program(kVS,kFS);
        if(!sProg){ if constexpr(Settings::debugLogging) LUCHS_LOG_HOST("[UI/Pfau][HM] ERROR: tiles program==0"); return; }
        uScale  = glGetUniformLocation(sProg,"uScale");
        uOffset = glGetUniformLocation(sProg,"uOffset");
    }
    if(!sVAO){ glGenVertexArrays(1,&sVAO); glGenBuffers(1,&sVBO); }

    if(!sPanelProg){
        sPanelProg = program(kPanelVS, kPanelFS);
        if(!sPanelProg){ if constexpr(Settings::debugLogging) LUCHS_LOG_HOST("[UI/Pfau][HM] ERROR: panel program==0"); }
        uViewportPx = glGetUniformLocation(sPanelProg, "uViewportPx");
        uPanelRectPx= glGetUniformLocation(sPanelProg, "uPanelRectPx");
        uRadiusPx   = glGetUniformLocation(sPanelProg, "uRadiusPx");
        uAlpha      = glGetUniformLocation(sPanelProg, "uAlpha");
        uBorderPx   = glGetUniformLocation(sPanelProg, "uBorderPx");
    }
    if(!sPanelVAO){ glGenVertexArrays(1,&sPanelVAO); glGenBuffers(1,&sPanelVBO); }

    float maxVal = 1e-6f;
    for(int i=0;i<nTiles;++i) maxVal = std::max(maxVal, entropy[i]+contrast[i]);

    std::vector<float> verts; verts.reserve(size_t(nTiles)*6u*3u);
    for(int y=0;y<tilesY;++y){
        for(int x=0;x<tilesX;++x){
            const float v = (entropy[y*tilesX+x]+contrast[y*tilesX+x]) / maxVal;
            const float px=float(x), py=float(y);
            const float q[18] = {
                px,py,v, px+1,py,v, px+1,py+1,v,
                px,py,v, px+1,py+1,v, px,py+1,v
            };
            verts.insert(verts.end(), q, q+18);
        }
    }

    // Pfau-Layout (Top-Right): identische Top-Margin wie Warzenschwein
    constexpr int contentHPx = 100;
    const float aspect = tilesY>0 ? float(tilesX)/float(tilesY) : 1.0f;
    const int   contentWPx = std::max(1, int(std::round(contentHPx*aspect)));
    const int panelW = contentWPx + int(2*Pfau::UI_PADDING);
    const int panelH = contentHPx + int(2*Pfau::UI_PADDING);
    const int panelX1 = width  - HeatmapOverlay::snapToPixel(Pfau::UI_MARGIN);
    const int panelX0 = panelX1 - panelW;
    const int panelY0 = HeatmapOverlay::snapToPixel(Pfau::UI_MARGIN);
    const int panelY1 = panelY0 + panelH;
    const int contentX0 = panelX0 + int(Pfau::UI_PADDING);
    const int contentY0 = panelY0 + int(Pfau::UI_PADDING);

    const float sx = ( (float)contentWPx / (float)width  / (float)tilesX ) * 2.0f;
    const float sy = ( (float)contentHPx / (float)height / (float)tilesY ) * 2.0f;
    const int   gridBottomY = (int)((height - (contentY0 + contentHPx)));
    const float ox = ( (float)contentX0 / (float)width )  * 2.0f - 1.0f;
    const float oy = ( (float)gridBottomY / (float)height ) * 2.0f - 1.0f;

    // State sichern
    GLint prevVAO=0, prevBuf=0, prevProg=0, srcRGB=0,dstRGB=0,srcA=0,dstA=0;
    GLboolean wasBlend=GL_FALSE, wasDepth=GL_FALSE;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING,&prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&prevBuf);
    glGetIntegerv(GL_CURRENT_PROGRAM,&prevProg);
    glGetBooleanv(GL_BLEND,&wasBlend);
    glGetBooleanv(GL_DEPTH_TEST,&wasDepth);
    glGetIntegerv(GL_BLEND_SRC_RGB,&srcRGB); glGetIntegerv(GL_BLEND_DST_RGB,&dstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA,&srcA); glGetIntegerv(GL_BLEND_DST_ALPHA,&dstA);

    // Panel (semi-transparent, rounded)
    {
        const float base[3]={0.10f,0.10f,0.10f};
        const float quad[30]={
            (float)panelX0,(float)panelY0, base[0],base[1],base[2],
            (float)panelX1,(float)panelY0, base[0],base[1],base[2],
            (float)panelX1,(float)panelY1, base[0],base[1],base[2],
            (float)panelX0,(float)panelY0, base[0],base[1],base[2],
            (float)panelX1,(float)panelY1, base[0],base[1],base[2],
            (float)panelX0,(float)panelY1, base[0],base[1],base[2],
        };
        glUseProgram(sPanelProg);
        if(uViewportPx>=0) glUniform2f(uViewportPx,(float)width,(float)height);
        if(uPanelRectPx>=0) glUniform4f(uPanelRectPx,(float)panelX0,(float)panelY0,(float)panelX1,(float)panelY1);
        if(uRadiusPx>=0)    glUniform1f(uRadiusPx,Pfau::UI_RADIUS);
        if(uAlpha>=0)       glUniform1f(uAlpha,Pfau::PANEL_ALPHA);
        if(uBorderPx>=0)    glUniform1f(uBorderPx,Pfau::UI_BORDER - 8.0f);
        glBindVertexArray(sPanelVAO);
        glBindBuffer(GL_ARRAY_BUFFER,sPanelVBO);
        glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)sizeof(quad),nullptr,GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER,0,(GLsizeiptr)sizeof(quad),quad);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(2*sizeof(float)));
        glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES,0,6);
    }

    // Tiles (opak) über Panel
    {
        glUseProgram(sProg);
        glUniform2f(uScale,sx,sy);
        glUniform2f(uOffset,ox,oy);
        glBindVertexArray(sVAO);
        glBindBuffer(GL_ARRAY_BUFFER,sVBO);
        const GLsizeiptr bytes = (GLsizeiptr)(verts.size()*sizeof(float));
        glBufferData(GL_ARRAY_BUFFER,bytes,nullptr,GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER,0,bytes,verts.data());
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)(2*sizeof(float)));
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES,0,(GLsizei)(verts.size()/3));
    }

    // Restore
    glBlendFuncSeparate(srcRGB,dstRGB,srcA,dstA);
    if(!wasBlend) glDisable(GL_BLEND);
    if(wasDepth) glEnable(GL_DEPTH_TEST);
    glDisableVertexAttribArray(0); glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER,prevBuf);
    glBindVertexArray((GLuint)prevVAO);
    glUseProgram((GLuint)prevProg);

    if constexpr(Settings::debugLogging){
        LUCHS_LOG_HOST("[UI/Pfau][HM] ok TR tiles=%dx%d verts=%zu zoom=%.6f c=(%.9f,%.9f) panel=[%d,%d..%d,%d]",
                       tilesX,tilesY,verts.size()/3,RS_ZOOM(ctx),RS_OFFSET_X(ctx),RS_OFFSET_Y(ctx),
                       panelX0,panelY0,panelX1,panelY1);
        GLenum err=glGetError(); if(err!=GL_NO_ERROR) LUCHS_LOG_HOST("[UI/Pfau][HM-GL] err=0x%04X",err);
    }
}

} // namespace HeatmapOverlay

#pragma warning(pop)
