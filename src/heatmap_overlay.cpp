///// Otter: GPU Heatmap overlay (fragment shader alpha) + Z0 sticker (argmax→CoM bend), no blur.
///// Schneefuchs: Coordinates harmonized with Eule; header/source kept in sync; no extra programs.
///// Maus: One-line ASCII logs; forwards blended interest to RendererState (screen coords); no zoom/pan changes.
///// Datei: src/heatmap_overlay.cpp

#include "pch.hpp"
#include "heatmap_overlay.hpp"
#include "settings.hpp"
#include "renderer_state.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_shaders.hpp"
#include "ui_gl.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace HeatmapOverlay {

// --- GL handles / uniforms (kompakt) -----------------------------------------
static GLuint sPanelVAO=0, sPanelVBO=0, sPanelProg=0;
static GLint  uViewportPx=-1, uPanelRectPx=-1, uRadiusPx=-1, uAlpha=-1, uBorderPx=-1;

static GLuint sHeatVAO=0, sHeatVBO=0, sHeatProg=0, sHeatTex=0;
static GLint  uHViewportPx=-1, uHContentRectPx=-1, uHGridTex=-1, uHAlphaBase=-1;
static GLint  uHMarkEnable=-1, uHMarkCenterPx=-1, uHMarkRadiusPx=-1, uHMarkAlpha=-1;

static int    sTexW=0, sTexH=0;
static float  sExposureEMA = 0.0f;

// Tuning (lokal, Settings bleiben schmal)
static constexpr float  kExposureDecay = 0.92f;
static constexpr float  kValueFloor    = 0.03f;
static constexpr double kBendStartZ    = 1.0;
static constexpr double kBendFullZ     = 18.0;
static constexpr double kBendExp       = 0.5;

// --- API ---------------------------------------------------------------------
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
    uHViewportPx=uHContentRectPx=uHGridTex=uHAlphaBase=-1;
    uHMarkEnable=uHMarkCenterPx=uHMarkRadiusPx=uHMarkAlpha=-1;

    sTexW=sTexH=0; sExposureEMA=0.0f;
}

void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height, int tileSize,
                 [[maybe_unused]] unsigned int textureId,
                 RendererState& ctx)
{
    ctx.interest.valid = false;
    if(!ctx.heatmapOverlayEnabled || width<=0||height<=0||tileSize<=0) return;

    const int tilesX = (width  + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const int nTiles = tilesX * tilesY;
    if((int)entropy.size()<nTiles || (int)contrast.size()<nTiles) return;

    // Programs & VAOs
    if(!sPanelProg){
        sPanelProg = UiGL::makeProgram(HeatmapShaders::PanelVS, HeatmapShaders::PanelFS);
        if(!sPanelProg){ if constexpr(Settings::debugLogging) LUCHS_LOG_HOST("[UI/Pfau][HM] panel program==0"); return; }
        uViewportPx = glGetUniformLocation(sPanelProg, "uViewportPx");
        uPanelRectPx= glGetUniformLocation(sPanelProg, "uPanelRectPx");
        uRadiusPx   = glGetUniformLocation(sPanelProg, "uRadiusPx");
        uAlpha      = glGetUniformLocation(sPanelProg, "uAlpha");
        uBorderPx   = glGetUniformLocation(sPanelProg, "uBorderPx");
    }
    if(!sHeatProg){
        sHeatProg = UiGL::makeProgram(HeatmapShaders::HeatVS, HeatmapShaders::HeatFS);
        if(!sHeatProg){ if constexpr(Settings::debugLogging) LUCHS_LOG_HOST("[UI/Pfau][HM] heat program==0"); return; }
        uHViewportPx    = glGetUniformLocation(sHeatProg, "uViewportPx");
        uHContentRectPx = glGetUniformLocation(sHeatProg, "uContentRectPx");
        uHGridTex       = glGetUniformLocation(sHeatProg, "uGrid");
        uHAlphaBase     = glGetUniformLocation(sHeatProg, "uAlphaBase");
        uHMarkEnable    = glGetUniformLocation(sHeatProg, "uMarkEnable");
        uHMarkCenterPx  = glGetUniformLocation(sHeatProg, "uMarkCenterPx");
        uHMarkRadiusPx  = glGetUniformLocation(sHeatProg, "uMarkRadiusPx");
        uHMarkAlpha     = glGetUniformLocation(sHeatProg, "uMarkAlpha");
    }
    UiGL::ensurePanelVAO(sPanelVAO, sPanelVBO);
    UiGL::ensureHeatVAO (sHeatVAO,  sHeatVBO);

    if(!sHeatTex){ glGenTextures(1,&sHeatTex); glBindTexture(GL_TEXTURE_2D,sHeatTex);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    }

    // Stabilized grid + near bias scoring
    float  currentMax = 1e-6f;
    int    bestIdx    = 0;
    float  bestRaw    = -1e30f;
    double bestScore  = -1e300;
    static thread_local std::vector<float> grid; grid.assign((size_t)nTiles, 0.0f);

    const double sigmaNdc = Settings::TargetBias::sigmaNdc;
    const double mix      = Settings::TargetBias::mix;
    const bool   nearOn   = Settings::TargetBias::enabled && (mix > 0.0);

    for(int i=0;i<nTiles;++i){
        const float raw = entropy[(size_t)i] + contrast[(size_t)i];
        if (raw > currentMax) currentMax = raw;
        grid[(size_t)i] = raw;
        double score = (double)raw;
        if (nearOn) {
            const int tx = i % tilesX, ty = i / tilesX;
            const double cx = ((double)tx + 0.5) * (double)width  / (double)tilesX;
            const double cy = ((double)ty + 0.5) * (double)height / (double)tilesY;
            const double ndcX = (cx / (double)width) * 2.0 - 1.0;
            const double ndcY = 1.0 - (cy / (double)height) * 2.0;
            const double r2   = ndcX*ndcX + ndcY*ndcY;
            const double w    = std::exp(- r2 / (sigmaNdc*sigmaNdc));
            score = (1.0 - mix) * score + mix * score * w;
        }
        if(score > bestScore){ bestScore=score; bestIdx=i; if(raw>bestRaw) bestRaw=raw; }
    }

    // Normalize 0..1 + floor; update EMA
    const float emaDecay = (sExposureEMA<=0.0f) ? currentMax : std::max(currentMax, kExposureDecay*sExposureEMA);
    sExposureEMA = std::max(1e-6f, emaDecay);
    for(int i=0;i<nTiles;++i){
        float v = grid[(size_t)i] / sExposureEMA;
        v = std::clamp(v + kValueFloor, 0.0f, 1.0f);
        grid[(size_t)i] = v;
    }

    glBindTexture(GL_TEXTURE_2D, sHeatTex);
    if(sTexW!=tilesX || sTexH!=tilesY){
        sTexW=tilesX; sTexH=tilesY;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, sTexW, sTexH, 0, GL_RED, GL_FLOAT, grid.data());
    }else{
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sTexW, sTexH, GL_RED, GL_FLOAT, grid.data());
    }

    // Layout
    constexpr int contentHPx = 92;
    const float aspect  = tilesY>0 ? float(tilesX)/float(tilesY) : 1.0f;
    const int   contentWPx = std::max(1, (int)std::round(contentHPx*aspect));
    const float sPanelScale = std::clamp(std::min(contentWPx,contentHPx)/160.0f, 0.60f, 1.0f);
    const int padPx = snapToPixel(Pfau::UI_PADDING * 0.75f);
    const int panelW = contentWPx + padPx*2, panelH = contentHPx + padPx*2;
    const int panelX1 = width  - snapToPixel(Pfau::UI_MARGIN);
    const int panelX0 = panelX1 - panelW;
    const int panelY0 = snapToPixel(Pfau::UI_MARGIN);
    const int panelY1 = panelY0 + panelH;
    const int contentX0 = panelX0 + padPx, contentY0 = panelY0 + padPx;
    const int contentX1 = contentX0 + contentWPx, contentY1 = contentY0 + contentHPx;

    // Save GL blend/program/vao state (kurz)
    GLint prevVAO=0, prevBuf=0, prevProg=0, srcRGB=0,dstRGB=0,srcA=0,dstA=0;
    GLboolean wasBlend=GL_FALSE, wasDepth=GL_FALSE;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING,&prevVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&prevBuf);
    glGetIntegerv(GL_CURRENT_PROGRAM,&prevProg);
    glGetBooleanv(GL_BLEND,&wasBlend);
    glGetBooleanv(GL_DEPTH_TEST,&wasDepth);
    glGetIntegerv(GL_BLEND_SRC_RGB,&srcRGB); glGetIntegerv(GL_BLEND_DST_RGB,&dstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA,&srcA); glGetIntegerv(GL_BLEND_DST_ALPHA,&dstA);

    // Panel
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
        const float panelAlpha = std::min(1.0f, Pfau::PANEL_ALPHA * 0.86f);
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
        glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES,0,6);
    }

    // Heat + Z0 sticker
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

        float alphaBase = std::min(1.0f, Pfau::PANEL_ALPHA * 1.10f);
        if(uHAlphaBase>=0)     glUniform1f(uHAlphaBase, alphaBase);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sHeatTex);
        if(uHGridTex>=0) glUniform1i(uHGridTex, 0);

        const int bx = bestIdx % tilesX, by = bestIdx / tilesX;

        // lokale 3x3-CoM
        double cx_com = bx + 0.5, cy_com = by + 0.5;
        {
            const int r = 1; const double sigma2 = 0.75*0.75; const double gammaW = 3.0;
            double wsum=0.0, xsum=0.0, ysum=0.0;
            for (int dy=-r; dy<=r; ++dy){
                const int ty = by + dy; if (ty < 0 || ty >= tilesY) continue;
                for (int dx=-r; dx<=r; ++dx){
                    const int tx = bx + dx; if (tx < 0 || tx >= tilesX) continue;
                    const size_t idx = (size_t)ty*(size_t)tilesX + (size_t)tx;
                    const double v = (double)grid[idx];
                    const double g = std::exp(-(dx*dx + dy*dy)/(2.0*sigma2));
                    const double w = std::pow(std::max(0.0, v), gammaW) * g;
                    wsum += w; xsum += w * (tx + 0.5); ysum += w * (ty + 0.5);
                }
            }
            if (wsum > 1e-9) { cx_com = xsum / wsum; cy_com = ysum / wsum; }
        }

        double bendT = 0.0;
        { const double z=(double)ctx.zoom;
          if (z > kBendStartZ) bendT = std::min(1.0, (z-kBendStartZ)/(kBendFullZ-kBendStartZ));
          bendT = std::pow(bendT, kBendExp);
        }

        const double cx_tile = (1.0 - bendT)*(bx + 0.5) + bendT*cx_com;
        const double cy_tile = (1.0 - bendT)*(by + 0.5) + bendT*cy_com;

        const float tileWPx_panel = (float)(contentX1 - contentX0) / std::max(1, tilesX);
        const float tileHPx_panel = (float)(contentY1 - contentY0) / std::max(1, tilesY);
        const float centerPxX_panel = (float)contentX0 + (float)cx_tile * tileWPx_panel;
        const float centerPxY_panel = (float)contentY0 + (float)cy_tile * tileHPx_panel;
        const float ringRpx_panel   = 0.70f * 0.5f * std::sqrt(tileWPx_panel*tileWPx_panel + tileHPx_panel*tileHPx_panel);

        if(uHMarkEnable>=0)   glUniform1f(uHMarkEnable,   1.0f);
        if(uHMarkCenterPx>=0) glUniform2f(uHMarkCenterPx, centerPxX_panel, centerPxY_panel);
        if(uHMarkRadiusPx>=0) glUniform1f(uHMarkRadiusPx, ringRpx_panel);
        if(uHMarkAlpha>=0)    glUniform1f(uHMarkAlpha,    0.95f);

        // screen-space Interest
        const float tileWPx_screen  = (float)width  / std::max(1, tilesX);
        const float tileHPx_screen  = (float)height / std::max(1, tilesY);
        const float centerPxX_screen= (float)cx_tile * tileWPx_screen;
        const float centerPxY_screen= (float)cy_tile * tileHPx_screen;
        const float ringRpx_screen  = 0.70f * 0.5f * std::sqrt(tileWPx_screen*tileWPx_screen + tileHPx_screen*tileHPx_screen);

        const double ndcX = (centerPxX_screen / (double)width)  * 2.0 - 1.0;
        const double ndcY = 1.0 - (centerPxY_screen / (double)height) * 2.0;
        const double rNdc = 0.5 * (((double)ringRpx_screen / (double)width ) * 2.0
                                 + ((double)ringRpx_screen / (double)height) * 2.0);

        ctx.interest.ndcX=ndcX; ctx.interest.ndcY=ndcY; ctx.interest.radiusNdc=rNdc;
        ctx.interest.strength=1.0; ctx.interest.valid=true;

        glBindVertexArray(sHeatVAO);
        glBindBuffer(GL_ARRAY_BUFFER,sHeatVBO);
        glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)sizeof(quad),nullptr,GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER,0,(GLsizeiptr)sizeof(quad),quad);
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES,0,6);

        if constexpr(Settings::debugLogging){
            const int bx0 = bestIdx % tilesX, by0 = bestIdx / tilesX;
            LUCHS_LOG_HOST("[ZSIG0] grid=%dx%d best=%d rawMax=%.6f nearMix=%.2f sig=%.2f ndc=(%.6f,%.6f)",
                           tilesX,tilesY,bestIdx,bestRaw,Settings::TargetBias::mix,Settings::TargetBias::sigmaNdc, ndcX, ndcY);
        }
    }

    // Restore GL state
    glBlendFuncSeparate(srcRGB,dstRGB,srcA,dstA);
    if(!wasBlend) glDisable(GL_BLEND);
    if(wasDepth) glEnable(GL_DEPTH_TEST);
    glBindBuffer(GL_ARRAY_BUFFER,prevBuf);
    glBindVertexArray((GLuint)prevVAO);
    glUseProgram((GLuint)prevProg);
}

} // namespace HeatmapOverlay
