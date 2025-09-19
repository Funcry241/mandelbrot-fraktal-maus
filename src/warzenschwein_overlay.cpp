///// Otter: HUD overlay; zoom/offset/FPS/entropy in deterministic layout.
///// Schneefuchs: No duplicate includes; API stable with header.
///// Maus: ASCII-only; minimal allocations per frame.
///// Datei: src/warzenschwein_overlay.cpp

#pragma warning(push)
#pragma warning(disable: 4100)

#include "pch.hpp"
#include "warzenschwein_overlay.hpp"
#include "warzenschwein_fontdata.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace WarzenschweinOverlay {

constexpr int glyphW=8, glyphH=12;

static GLuint vao=0, vbo=0, prog=0;
static std::vector<float> verts;    // x,y,r,g,b
static std::vector<float> panel;    // x,y,r,g,b (nur als Träger)
static std::string text;
static bool visible=true;

static GLint uViewport=-1,uScaleLoc=-1,uAlpha=-1,uRect=-1,uRadius=-1,uBorder=-1,uMode=-1;

static const char* vs = R"GLSL(
#version 430 core
layout(location=0) in vec2 aPosPx; layout(location=1) in vec3 aColor;
uniform vec2 uViewport; uniform vec2 uHudScale; out vec3 vCol; out vec2 vPx;
vec2 toNdc(vec2 p, vec2 vp){ return vec2(p.x/vp.x*2-1, 1-p.y/vp.y*2); }
void main(){ vec2 s=aPosPx*uHudScale; vPx=s; vCol=aColor; gl_Position=vec4(toNdc(s,uViewport),0,1); }
)GLSL";

static const char* fs = R"GLSL(
#version 430 core
in vec3 vCol; in vec2 vPx; out vec4 FragColor;
uniform vec2 uViewport; uniform float uAlpha; uniform vec4 uRect; uniform float uRadius,uBorder; uniform int uMode;
float sdRoundRect(vec2 p, vec2 b, float r){ vec2 d=abs(p)-b+vec2(r); return length(max(d,0))-r; }
void main(){
  if(uMode==1){
    vec2 c=0.5*(uRect.xy+uRect.zw), b=0.5*(uRect.zw-uRect.xy);
    float d=sdRoundRect(vPx-c,b,uRadius); if(d>0) discard;
    float edge=smoothstep(0,1,1.0-clamp(d+uBorder,0.0,uBorder)/max(uBorder,1e-6));
    vec3 col=mix(vCol, vec3(1.0,0.82,0.32), 0.20*edge); FragColor=vec4(col,uAlpha);
  } else {
    FragColor=vec4(vCol,1.0);
  }
}
)GLSL";

static GLuint make(GLenum t, const char* src){
    GLuint s=glCreateShader(t); if(!s) return 0;
    glShaderSource(s,1,&src,nullptr); glCompileShader(s);
    GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ if constexpr(Settings::debugLogging){ char log[1024]{}; glGetShaderInfoLog(s,1024,nullptr,log);
        LUCHS_LOG_HOST("[UI/Pfau][WS] shader compile fail: %s", log);} glDeleteShader(s); return 0; }
    return s;
}

// NOTE: Lokale Variablennamen bewusst NICHT "vs"/"fs", um C4459 (shadowing) zu vermeiden.
static GLuint program(const char* v, const char* f){
    GLuint shVS=make(GL_VERTEX_SHADER,v), shFS=make(GL_FRAGMENT_SHADER,f);
    if(!shVS||!shFS){ if(shVS)glDeleteShader(shVS); if(shFS)glDeleteShader(shFS); return 0; }
    GLuint p=glCreateProgram(); if(!p){ glDeleteShader(shVS); glDeleteShader(shFS); return 0; }
    glAttachShader(p,shVS); glAttachShader(p,shFS); glLinkProgram(p);
    glDeleteShader(shVS); glDeleteShader(shFS);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ if constexpr(Settings::debugLogging){ char log[1024]{}; glGetProgramInfoLog(p,1024,nullptr,log);
        LUCHS_LOG_HOST("[UI/Pfau][WS] program link fail: %s", log);} glDeleteProgram(p); return 0; }
    return p;
}

static void initGL(){
    if(vao) return;
    prog = program(vs,fs);
    if(!prog){ if constexpr(Settings::debugLogging) LUCHS_LOG_HOST("[UI/Pfau][WS] initGL fail (prog=0)"); return; }
    glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo);
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(2*sizeof(float))); glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    uViewport=glGetUniformLocation(prog,"uViewport");
    uScaleLoc=glGetUniformLocation(prog,"uHudScale");
    uAlpha   =glGetUniformLocation(prog,"uAlpha");
    uRect    =glGetUniformLocation(prog,"uRect");
    uRadius  =glGetUniformLocation(prog,"uRadius");
    uBorder  =glGetUniformLocation(prog,"uBorder");
    uMode    =glGetUniformLocation(prog,"uMode");
}

static void buildPanel(std::vector<float>& out, float x0,float y0,float x1,float y1){
    const float bg[3]={0.10f,0.10f,0.10f};
    const float q[30]={ x0,y0,bg[0],bg[1],bg[2], x1,y0,bg[0],bg[1],bg[2],
                        x1,y1,bg[0],bg[1],bg[2], x0,y0,bg[0],bg[1],bg[2],
                        x1,y1,bg[0],bg[1],bg[2], x0,y1,bg[0],bg[1],bg[2] };
    out.insert(out.end(), q, q+30);
}

void generateOverlayQuads(const std::string& t, int viewportW, int viewportH, float zoom,
                          std::vector<float>& vOut, std::vector<float>& pOut) {
    (void)viewportW; (void)viewportH; (void)zoom;
    vOut.clear(); pOut.clear();
    const float scalePx = std::max(1.0f, Settings::hudPixelSize);

    // Außenabstand soll EXAKT UI_MARGIN sein -> Content-Anker = MARGIN + PADDING
    const float marginX = Pfau::UI_MARGIN,  marginY = Pfau::UI_MARGIN;
    const float pad     = Pfau::UI_PADDING;
    const float x0 = (float)WarzenschweinOverlay::snapToPixel(marginX + pad);
    const float y0 = (float)WarzenschweinOverlay::snapToPixel(marginY + pad);

    // Zeilen splitten
    std::vector<std::string> lines; { std::string cur; cur.reserve(64);
        for(char c: t){ if(c=='\n'){ lines.push_back(cur); cur.clear(); } else cur+=c; }
        if(!cur.empty()) lines.push_back(cur);
    }

    // Content-Box
    size_t maxW=0; for(const auto& l:lines) maxW=std::max(maxW,l.size());
    const float advX=(glyphW+1)*scalePx, advY=(glyphH+2)*scalePx;
    const float boxW=float(maxW)*advX, boxH=float(lines.size())*advY;

    // Panel-Rand exakt bei UI_MARGIN
    buildPanel(pOut, x0 - pad, y0 - pad, x0 + boxW + pad, y0 + boxH + pad);

    // Glyphen
    const float r=1.0f,g=0.82f,b=0.32f;
    for(size_t row=0; row<lines.size(); ++row){
        const std::string& line=lines[row];
        const float yBase=y0+row*advY;
        for(size_t col=0; col<line.size(); ++col){
            const auto& glyph=WarzenschweinFont::get(line[col]);
            const float xBase=x0+col*advX;
            for(int gy=0; gy<glyphH; ++gy){
                const uint8_t bits=glyph[gy];
                for(int gx=0; gx<glyphW; ++gx){
                    if((bits>>(7-gx))&1){
                        const float x=xBase+gx*scalePx, y=yBase+gy*scalePx;
                        const float q[30]={ x,y,r,g,b, x+scalePx,y,r,g,b,
                                            x+scalePx,y+scalePx,r,g,b, x,y,r,g,b,
                                            x+scalePx,y+scalePx,r,g,b, x,y+scalePx,r,g,b };
                        vOut.insert(vOut.end(), q, q+30);
                    }
                }
            }
        }
    }
}

void drawOverlay(float zoom){
    if(!Settings::warzenschweinOverlayEnabled || !visible || text.empty()) return;

    GLint vp[4]={0,0,0,0}; glGetIntegerv(GL_VIEWPORT,vp);
    const int vpW=vp[2], vpH=vp[3];

    initGL(); if(!prog) return;
    generateOverlayQuads(text, vpW, vpH, zoom, verts, panel);

    float xMin= std::numeric_limits<float>::max(), yMin=xMin, xMax=-xMin, yMax=-yMin;
    for(size_t i=0;i+4<panel.size();i+=5){ xMin=std::min(xMin,panel[i]); yMin=std::min(yMin,panel[i+1]);
                                           xMax=std::max(xMax,panel[i]); yMax=std::max(yMax,panel[i+1]); }

    // State sichern
    GLint prevVAO=0, prevBuf=0, prevProg=0, srcRGB=0,dstRGB=0,srcA=0,dstA=0;
    GLboolean wasBlend=GL_FALSE, wasDepth=GL_FALSE;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING,&prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&prevBuf);
    glGetIntegerv(GL_CURRENT_PROGRAM,&prevProg);
    glGetBooleanv(GL_BLEND,&wasBlend); glGetBooleanv(GL_DEPTH_TEST,&wasDepth);
    glGetIntegerv(GL_BLEND_SRC_RGB,&srcRGB); glGetIntegerv(GL_BLEND_DST_RGB,&dstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA,&srcA); glGetIntegerv(GL_BLEND_DST_ALPHA,&dstA);

    glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(prog); glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER,vbo);
    if(uViewport>=0) glUniform2f(uViewport,(float)vpW,(float)vpH);
    if(uScaleLoc>=0) glUniform2f(uScaleLoc,1.0f,1.0f);

    // Pass 1: Panel
    if(!panel.empty()){
        if(uMode>=0)   glUniform1i(uMode,1);
        if(uAlpha>=0)  glUniform1f(uAlpha,Pfau::PANEL_ALPHA);
        if(uRect>=0)   glUniform4f(uRect,xMin,yMin,xMax,yMax);
        if(uRadius>=0) glUniform1f(uRadius,Pfau::UI_RADIUS);
        if(uBorder>=0) glUniform1f(uBorder,Pfau::UI_BORDER);
        const GLsizeiptr bytes=(GLsizeiptr)(panel.size()*sizeof(float));
        glBufferData(GL_ARRAY_BUFFER,bytes,nullptr,GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER,0,bytes,panel.data());
        glDrawArrays(GL_TRIANGLES,0,(GLsizei)(panel.size()/5));
    }
    // Pass 2: Text
    if(!verts.empty()){
        if(uMode>=0)   glUniform1i(uMode,0);
        if(uAlpha>=0)  glUniform1f(uAlpha,1.0f);
        const GLsizeiptr bytes=(GLsizeiptr)(verts.size()*sizeof(float));
        glBufferData(GL_ARRAY_BUFFER,bytes,nullptr,GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER,0,bytes,verts.data());
        glDrawArrays(GL_TRIANGLES,0,(GLsizei)(verts.size()/5));
    }

    // Restore
    glBlendFuncSeparate(srcRGB,dstRGB,srcA,dstA);
    if(!wasBlend) glDisable(GL_BLEND);
    if(wasDepth)  glEnable(GL_DEPTH_TEST);
    glBindBuffer(GL_ARRAY_BUFFER,prevBuf);
    glBindVertexArray((GLuint)prevVAO);
    glUseProgram((GLuint)prevProg);

    if constexpr(Settings::debugLogging){
        GLenum err=glGetError(); if(err!=GL_NO_ERROR) LUCHS_LOG_HOST("[UI/Pfau][WS-GL] err=0x%04X",err);
    }
}

void toggle(){ visible=!visible; }
void setText(const std::string& t){ text=t; }

void cleanup(){
    if(vao) glDeleteVertexArrays(1,&vao);
    if(vbo) glDeleteBuffers(1,&vbo);
    if(prog) glDeleteProgram(prog);
    vao=vbo=prog=0; verts.clear(); panel.clear(); text.clear(); visible=false;
    uViewport=uScaleLoc=uAlpha=uRect=uRadius=uBorder=uMode=-1;
}

} // namespace WarzenschweinOverlay

#pragma warning(pop)
