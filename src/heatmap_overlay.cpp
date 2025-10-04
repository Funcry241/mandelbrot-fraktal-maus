///// Otter: GPU Heatmap overlay (fragment shader alpha) + Z0 sticker (argmax), no blur.
/// /// Schneefuchs: Coordinates harmonized with Eule; header/source kept in sync; no extra programs.
/// /// Maus: One-line ASCII logs; forwards interest to RendererState (screen coords); no zoom/pan changes.
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

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------
static GLuint sPanelVAO=0, sPanelVBO=0, sPanelProg=0;     // Panel (rounded, alpha)
static GLint  uViewportPx=-1, uPanelRectPx=-1, uRadiusPx=-1, uAlpha=-1, uBorderPx=-1;

static GLuint sHeatVAO=0, sHeatVBO=0, sHeatProg=0, sHeatTex=0; // Heat (texture + alpha + sticker)
static GLint  uHViewportPx=-1, uHContentRectPx=-1, uHGridSize=-1, uHGridTex=-1, uHAlphaBase=-1;
// Z0 sticker uniforms inside Heat FS
static GLint  uHMarkEnable=-1, uHMarkCenterPx=-1, uHMarkRadiusPx=-1, uHMarkAlpha=-1;

static int    sTexW=0, sTexH=0;

// Exposure control (stabilized normalization)
static float  sExposureEMA = 0.0f;            // tracks a smoothed per-frame maximum
static constexpr float kExposureDecay = 0.92f; // EMA decay (0.90..0.95 recommended)
static constexpr float kValueFloor    = 0.03f; // minimal floor added before shader mapping

// -----------------------------------------------------------------------------
// Shaders
// -----------------------------------------------------------------------------
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
  return length(max(d,0.0)) - r;      // d<0 inside, d>0 outside
}

void main(){
  vec2 c = 0.5 * (uPanelRectPx.xy + uPanelRectPx.zw);
  vec2 b = 0.5 * (uPanelRectPx.zw - uPanelRectPx.xy);
  float d = sdRoundRect(vPx - c, b, uRadiusPx);

  float aa   = fwidth(d);
  float body = 1.0 - smoothstep(0.0, aa, max(d, 0.0));  // 1 inside, 0 outside

  float inner = smoothstep(-uBorderPx*0.5, 0.0, d);

  vec3 borderCol = vec3(1.0, 0.82, 0.32);
  vec3 col = mix(vColor, borderCol, 0.08 * inner); // subtle: 8% mix

  FragColor = vec4(col, uAlpha * body);
}
)GLSL";

// Heat pass: draw only the content rect; fragment shader samples the tiles texture
// (NO BLUR), maps to gold, emits alpha by value, and overlays a Z0 sticker.
static const char* kHeatVS = R"GLSL(#version 430 core
layout(location=0) in vec2 aPosPx;
uniform vec2 uViewportPx; out vec2 vPx;
vec2 toNdc(vec2 p, vec2 vp){ return vec2(p.x/vp.x*2-1, 1-p.y/vp.y*2); }
void main(){ vPx=aPosPx; gl_Position=vec4(toNdc(aPosPx,uViewportPx),0,1); }
)GLSL";

static const char* kHeatFS = R"GLSL(#version 430 core
in vec2 vPx; out vec4 FragColor;
uniform vec4 uContentRectPx;  // [x0,y0,x1,y1] inner area (no padding)
uniform ivec2 uGridSize;      // tilesX, tilesY (kept for compatibility)
uniform sampler2D uGrid;      // GL_R16F or GL_R32F, normalized to 0..1
uniform float uAlphaBase;     // base alpha (scales Pfau alpha)
// Z0 marker uniforms
uniform float uMarkEnable;    // 0.0/1.0
uniform vec2  uMarkCenterPx;  // screen pixel coords (panel space)
uniform float uMarkRadiusPx;  // ring radius in px
uniform float uMarkAlpha;     // marker intensity

vec3 mapGold(float v){
  float g = clamp(v,0.0,1.0);
  g = smoothstep(0.0,1.0,g);
  g = pow(g, 0.90); // a tad brighter than 0.94
  return mix(vec3(0.08,0.08,0.10), vec3(0.98,0.78,0.30), g);
}

void main(){
  vec2 sizePx = uContentRectPx.zw - uContentRectPx.xy;
  vec2 uv = (vPx - uContentRectPx.xy) / sizePx;

  // clip strictly to content rect (no alpha outside)
  if(any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))){
    FragColor = vec4(0.0); return;
  }

  // --- NO BLUR: single texture fetch ---
  float v = texture(uGrid, uv).r;

  // alpha from value
  float a = smoothstep(0.05, 0.65, v) * uAlphaBase;
  vec4 base = vec4(mapGold(v), a);

  // --- Z0 sticker overlay (ring + crosshair), analytic in pixel space ---
  float m = 0.0;
  if(uMarkEnable > 0.5){
    vec2  d2   = vPx - uMarkCenterPx;
    float r    = length(d2);
    float edge = abs(r - uMarkRadiusPx);
    float aa   = fwidth(edge) + 0.75;         // bias to stay visible
    float ring = 1.0 - smoothstep(1.5, 1.5+aa, edge);
    float cx   = 1.0 - smoothstep(0.6, 0.6+aa, abs(d2.x));
    float cy   = 1.0 - smoothstep(0.6, 0.6+aa, abs(d2.y));
    m = max(ring, max(cx, cy)) * uMarkAlpha;
  }
  vec3 markCol = vec3(0.60, 1.00, 0.60);
  vec4 outCol  = mix(base, vec4(markCol, 1.0), m);
  outCol.a     = max(base.a, max(outCol.a, m));
  FragColor    = outCol;
}
)GLSL";

// -----------------------------------------------------------------------------
// GL helpers
// -----------------------------------------------------------------------------
static GLuint make(GLenum type, const char* src){
    GLuint sh=glCreateShader(type); if(!sh) return 0;
    glShaderSource(sh,1,&src,nullptr); glCompileShader(sh);
    GLint ok=0; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){
        if constexpr(Settings::debugLogging){ char log[1024]{}; glGetShaderInfoLog(sh,1024,nullptr,log);
            LUCHS_LOG_HOST("[UI/Pfau][HM] shader compile fail: %s", log); }
        glDeleteShader(sh); return 0;
    }
    return sh;
}
static GLuint program(const char* vs, const char* fs){
    GLuint v=make(GL_VERTEX_SHADER,vs), f=make(GL_FRAGMENT_SHADER,fs);
    if(!v||!f){ if(v)glDeleteShader(v); if(f)glDeleteShader(f); return 0; }
    GLuint p=glCreateProgram(); if(!p){ glDeleteShader(v); glDeleteShader(f); return 0; }
    glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){
        if constexpr(Settings::debugLogging){ char log[1024]{}; glGetProgramInfoLog(p,1024,nullptr,log);
            LUCHS_LOG_HOST("[UI/Pfau][HM] program link fail: %s", log); }
        glDeleteProgram(p); return 0;
    }
    return p;
}

// -----------------------------------------------------------------------------
// API
// -----------------------------------------------------------------------------
void toggle(RendererState& ctx){
#if defined(USE_HEATMAP_OVERLAY)
    ctx.heatmapOverlayEnabled = !ctx.heatmapOverlayEnabled;
#else
    (void)ctx;
#endif
}

void cleanup(){
    if(sPanelVAO) glDeleteVertexArrays(1,&sPanelVAO);
    if(sPanelVBO) glDeleteBuffers(1,&sPanelVBO);
    if(sPanelProg) glDeleteProgram(sPanelProg);
    sPanelVAO=sPanelVBO=sPanelProg=0;
    uViewportPx=uPanelRectPx=uRadiusPx=uAlpha=uBorderPx=-1;

    if(sHeatVAO) glDeleteVertexArrays(1,&sHeatVAO);
    if(sHeatVBO) glDeleteBuffers(1,&sHeatVBO);
    if(sHeatProg) glDeleteProgram(sHeatProg);
    if(sHeatTex) glDeleteTextures(1,&sHeatTex);
    sHeatVAO=sHeatVBO=sHeatProg=sHeatTex=0;
    uHViewportPx=uHContentRectPx=uHGridSize=uHGridTex=uHAlphaBase=-1;
    uHMarkEnable=uHMarkCenterPx=uHMarkRadiusPx=uHMarkAlpha=-1;

    sTexW=sTexH=0;
    sExposureEMA = 0.0f;
}

void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height, int tileSize,
                 [[maybe_unused]] unsigned int textureId,
                 RendererState& ctx)
{
    // Interest pro Frame invalidieren (nur gültig wenn wir gleich setzen)
    ctx.interest.valid = false;

    if(!ctx.heatmapOverlayEnabled || width<=0||height<=0||tileSize<=0) return;

    // Tiles geometry
    const int tilesX = (width  + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int nTiles = tilesX * tilesY;
    if((int)entropy.size()<nTiles || (int)contrast.size()<nTiles) return;

    // --- Lazy init programs ---
    if(!sPanelProg){
        sPanelProg = program(kPanelVS, kPanelFS);
        if(!sPanelProg){ if constexpr(Settings::debugLogging) LUCHS_LOG_HOST("[UI/Pfau][HM] ERROR: panel program==0"); return; }
        uViewportPx = glGetUniformLocation(sPanelProg, "uViewportPx");
        uPanelRectPx= glGetUniformLocation(sPanelProg, "uPanelRectPx");
        uRadiusPx   = glGetUniformLocation(sPanelProg, "uRadiusPx");
        uAlpha      = glGetUniformLocation(sPanelProg, "uAlpha");
        uBorderPx   = glGetUniformLocation(sPanelProg, "uBorderPx");
    }
    if(!sHeatProg){
        sHeatProg = program(kHeatVS, kHeatFS);
        if(!sHeatProg){ if constexpr(Settings::debugLogging) LUCHS_LOG_HOST("[UI/Pfau][HM] ERROR: heat program==0"); return; }
        uHViewportPx    = glGetUniformLocation(sHeatProg, "uViewportPx");
        uHContentRectPx = glGetUniformLocation(sHeatProg, "uContentRectPx");
        uHGridSize      = glGetUniformLocation(sHeatProg, "uGridSize");
        uHGridTex       = glGetUniformLocation(sHeatProg, "uGrid");
        uHAlphaBase     = glGetUniformLocation(sHeatProg, "uAlphaBase");
        // Z0 marker uniforms
        uHMarkEnable    = glGetUniformLocation(sHeatProg, "uMarkEnable");
        uHMarkCenterPx  = glGetUniformLocation(sHeatProg, "uMarkCenterPx");
        uHMarkRadiusPx  = glGetUniformLocation(sHeatProg, "uMarkRadiusPx");
        uHMarkAlpha     = glGetUniformLocation(sHeatProg, "uMarkAlpha");
    }

    if(!sPanelVAO){ glGenVertexArrays(1,&sPanelVAO); glGenBuffers(1,&sPanelVBO); }
    if(!sHeatVAO){  glGenVertexArrays(1,&sHeatVAO);  glGenBuffers(1,&sHeatVBO);  }

    if(!sHeatTex){ glGenTextures(1,&sHeatTex); glBindTexture(GL_TEXTURE_2D,sHeatTex);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    }

    // --- Build/Update tiles texture with stabilized exposure, track raw argmax ---
    float currentMax = 1e-6f;
    int   bestIdxRaw = 0;
    float bestRaw    = -1e30f;

    for(int i=0;i<nTiles;++i){
        const float raw = entropy[(size_t)i] + contrast[(size_t)i];
        if (raw > currentMax) currentMax = raw;
        if (raw > bestRaw){ bestRaw = raw; bestIdxRaw = i; }
    }

    // Initialize/Update EMA: keep from decreasing too fast (decay) but allow rise immediately
    if (sExposureEMA <= 0.0f) sExposureEMA = currentMax;
    else                      sExposureEMA = std::max(kExposureDecay * sExposureEMA, currentMax);

    const float normMax = std::max(1e-6f, sExposureEMA);

    std::vector<float> grid; grid.resize((size_t)nTiles);
    for(int i=0;i<nTiles;++i){
        float v = (entropy[(size_t)i] + contrast[(size_t)i]) / normMax;
        v = std::clamp(v + kValueFloor, 0.0f, 1.0f); // bring-up floor (pre-smoothstep)
        grid[(size_t)i] = v;
    }

    glBindTexture(GL_TEXTURE_2D, sHeatTex);
    if(sTexW!=tilesX || sTexH!=tilesY){
        sTexW=tilesX; sTexH=tilesY;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, sTexW, sTexH, 0, GL_RED, GL_FLOAT, grid.data());
    }else{
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sTexW, sTexH, GL_RED, GL_FLOAT, grid.data());
    }

    // --- Layout (Pfau TR) ---
    constexpr int contentHPx = 92; // optisch leichter als 100
    const float aspect  = tilesY>0 ? float(tilesX)/float(tilesY) : 1.0f;
    const int   contentWPx = std::max(1, (int)std::round(contentHPx*aspect));

    const float sPanelScale = std::clamp(std::min(contentWPx,contentHPx)/160.0f, 0.60f, 1.0f);
    const int padPx = snapToPixel(Pfau::UI_PADDING * 0.75f);

    const int panelW = contentWPx + padPx*2;
    const int panelH = contentHPx + padPx*2;

    const int panelX1 = width  - snapToPixel(Pfau::UI_MARGIN);
    const int panelX0 = panelX1 - panelW;
    const int panelY0 = snapToPixel(Pfau::UI_MARGIN);
    const int panelY1 = panelY0 + panelH;

    const int contentX0 = panelX0 + padPx;
    const int contentY0 = panelY0 + padPx;
    const int contentX1 = contentX0 + contentWPx;
    const int contentY1 = contentY0 + contentHPx;

    // --- Save GL state ---
    GLint prevVAO=0, prevBuf=0, prevProg=0, srcRGB=0,dstRGB=0,srcA=0,dstA=0;
    GLboolean wasBlend=GL_FALSE, wasDepth=GL_FALSE;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING,&prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&prevBuf);
    glGetIntegerv(GL_CURRENT_PROGRAM,&prevProg);
    glGetBooleanv(GL_BLEND,&wasBlend);
    glGetBooleanv(GL_DEPTH_TEST,&wasDepth);
    glGetIntegerv(GL_BLEND_SRC_RGB,&srcRGB); glGetIntegerv(GL_BLEND_DST_RGB,&dstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA,&srcA); glGetIntegerv(GL_BLEND_DST_ALPHA,&dstA);

    // --- Panel (rounded, semi-transparent) ---
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

        const float panelAlpha = Pfau::PANEL_ALPHA * 0.86f;   // ≈0.72 bei 0.84 Basis
        const float radiusPx   = Pfau::UI_RADIUS * (0.85f * sPanelScale);
        const float borderPx   = Pfau::UI_BORDER * (0.35f * sPanelScale);

        glUseProgram(sPanelProg);
        if(uViewportPx>=0) glUniform2f(uViewportPx,(float)width,(float)height);
        if(uPanelRectPx>=0) glUniform4f(uPanelRectPx,(float)panelX0,(float)panelY0,(float)panelX1,(float)panelY1);
        if(uRadiusPx>=0)    glUniform1f(uRadiusPx,radiusPx);
        if(uAlpha>=0)       glUniform1f(uAlpha,panelAlpha);
        if(uBorderPx>=0)    glUniform1f(uBorderPx,borderPx);

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

    // --- Heat pass (content rect quad; FS does value mapping + Z0 sticker) ---
    {
        const float quad[12]={
            (float)contentX0,(float)contentY0,
            (float)contentX1,(float)contentY0,
            (float)contentX1,(float)contentY1,
            (float)contentX0,(float)contentY0,
            (float)contentX1,(float)contentY1,
            (float)contentX0,(float)contentY1,
        };

        glUseProgram(sHeatProg);
        if(uHViewportPx>=0)    glUniform2f(uHViewportPx,(float)width,(float)height);
        if(uHContentRectPx>=0) glUniform4f(uHContentRectPx,(float)contentX0,(float)contentY0,(float)contentX1,(float)contentY1);
        if(uHGridSize>=0)      glUniform2i(uHGridSize, tilesX, tilesY);

        float alphaBase = Pfau::PANEL_ALPHA * 1.10f; // etwas höheres Grund-Alpha
        if (alphaBase > 1.0f) alphaBase = 1.0f;
        if(uHAlphaBase>=0)     glUniform1f(uHAlphaBase, alphaBase);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sHeatTex);
        if(uHGridTex>=0) glUniform1i(uHGridTex, 0);

        // --- Argmax: Panel-Coords (für Sticker) -------------------------------
        const int   bx = bestIdxRaw % tilesX;
        const int   by = bestIdxRaw / tilesX;
        const float tileWPx_panel = (float)(contentX1 - contentX0) / std::max(1, tilesX);
        const float tileHPx_panel = (float)(contentY1 - contentY0) / std::max(1, tilesY);
        const float centerPxX_panel = (float)contentX0 + (bx + 0.5f) * tileWPx_panel;
        const float centerPxY_panel = (float)contentY0 + (by + 0.5f) * tileHPx_panel;
        const float ringRpx_panel   = 0.70f * 0.5f * std::sqrt(tileWPx_panel*tileWPx_panel
                                                             + tileHPx_panel*tileHPx_panel);

        if(uHMarkEnable>=0)   glUniform1f(uHMarkEnable,   1.0f);
        if(uHMarkCenterPx>=0) glUniform2f(uHMarkCenterPx, centerPxX_panel, centerPxY_panel);
        if(uHMarkRadiusPx>=0) glUniform1f(uHMarkRadiusPx, ringRpx_panel);
        if(uHMarkAlpha>=0)    glUniform1f(uHMarkAlpha,    0.95f);

        // --- Argmax: Screen-Coords (für Zoom-Interest) -----------------------
        const float tileWPx_screen = (float)width  / std::max(1, tilesX);
        const float tileHPx_screen = (float)height / std::max(1, tilesY);
        const float centerPxX_screen = (bx + 0.5f) * tileWPx_screen;
        const float centerPxY_screen = (by + 0.5f) * tileHPx_screen;
        const float ringRpx_screen   = 0.70f * 0.5f * std::sqrt(tileWPx_screen*tileWPx_screen
                                                              + tileHPx_screen*tileHPx_screen);

        const double ndcX = (centerPxX_screen / (double)width)  * 2.0 - 1.0;
        const double ndcY = 1.0 - (centerPxY_screen / (double)height) * 2.0;
        const double rNdc = 0.5 * (((double)ringRpx_screen / (double)width ) * 2.0
                                 + ((double)ringRpx_screen / (double)height) * 2.0);

        ctx.interest.ndcX      = ndcX;
        ctx.interest.ndcY      = ndcY;
        ctx.interest.radiusNdc = rNdc;
        ctx.interest.strength  = 1.0;   // Z0 = Argmax voll gewichtet
        ctx.interest.valid     = true;

        glBindVertexArray(sHeatVAO);
        glBindBuffer(GL_ARRAY_BUFFER,sHeatVBO);
        glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)sizeof(quad),nullptr,GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER,0,(GLsizeiptr)sizeof(quad),quad);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(float),(void*)0);

        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES,0,6);

        if constexpr(Settings::debugLogging){
            const double ndcX_panel = (centerPxX_panel / (double)width)  * 2.0 - 1.0;
            const double ndcY_panel = 1.0 - (centerPxY_panel / (double)height) * 2.0;
            LUCHS_LOG_HOST("[ZSIG0] grid=%dx%d best=%d raw=%.6f ndc_screen=(%.6f,%.6f) ndc_panel=(%.6f,%.6f) Rpx_screen=%.2f",
                           tilesX, tilesY, bestIdxRaw, bestRaw, ndcX, ndcY, ndcX_panel, ndcY_panel, ringRpx_screen);
        }
    }

    // --- Restore GL state ---
    glBlendFuncSeparate(srcRGB,dstRGB,srcA,dstA);
    if(!wasBlend) glDisable(GL_BLEND);
    if(wasDepth) glEnable(GL_DEPTH_TEST);
    glDisableVertexAttribArray(0); glDisableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER,prevBuf);
    glBindVertexArray((GLuint)prevVAO);
    glUseProgram((GLuint)prevProg);

    if constexpr(Settings::debugLogging){
        LUCHS_LOG_HOST("[UI/Pfau][HM] ok TR grid=%dx%d zoom=%.6f c=(%.9f,%.9f) panel=[%d,%d..%d,%d] content=[%d,%d..%d,%d] expEMA=%.6f",
                       tilesX,tilesY,RS_ZOOM(ctx),RS_OFFSET_X(ctx),RS_OFFSET_Y(ctx),
                       panelX0,panelY0,panelX1,panelY1, contentX0,contentY0,contentX1,contentY1, sExposureEMA);
        GLenum err=glGetError(); if(err!=GL_NO_ERROR) LUCHS_LOG_HOST("[UI/Pfau][HM-GL] err=0x%04X",err);
    }
}

} // namespace HeatmapOverlay

#pragma warning(pop)
