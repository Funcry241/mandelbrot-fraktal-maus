#ifndef INCLUDE_STB_EASY_FONT_H
#define INCLUDE_STB_EASY_FONT_H

#include <stdlib.h>
#include <math.h>

// --- Unused-Attribut für Helper ---
#if defined(__GNUC__) || defined(__clang__)
#define STB_UNUSED __attribute__((unused))
#else
#define STB_UNUSED
#endif

// --- Daten-Tabellen (wie gehabt, gekürzt im Beispiel) ---
static struct stb_easy_font_info_struct { unsigned char advance, h_seg, v_seg; } stb_easy_font_charinfo[96] = { /* ... */ };
static unsigned char stb_easy_font_hseg[214] = { /* ... */ };
static unsigned char stb_easy_font_vseg[253] = { /* ... */ };

typedef struct { unsigned char c[4]; } stb_easy_font_color;

static int stb_easy_font_draw_segs(float x, float y, unsigned char *segs, int num_segs, int vertical, stb_easy_font_color c, char *vbuf, int vbuf_size, int offset)
{
    for (int i = 0; i < num_segs; ++i) {
        int len = segs[i] & 7;
        x += (float)((segs[i] >> 3) & 1);
        if (len && offset + 64 <= vbuf_size) {
            float y0 = y + (float)(segs[i] >> 4);
            for (int j = 0; j < 4; ++j) {
                *((float *)(vbuf+offset+0))  = x + (j==1 || j==2 ? (vertical ? 1 : len) : 0);
                *((float *)(vbuf+offset+4))  = y0 + (j >= 2 ? (vertical ? len : 1) : 0);
                *((float *)(vbuf+offset+8))  = 0.f;
                *((stb_easy_font_color *)(vbuf+offset+12)) = c;
                offset += 16;
            }
        }
    }
    return offset;
}

static float stb_easy_font_spacing_val = 0;
static void stb_easy_font_spacing(float spacing) STB_UNUSED;
static int stb_easy_font_width(char *text)   STB_UNUSED;
static int stb_easy_font_height(char *text)  STB_UNUSED;
static int stb_easy_font_print(float x, float y, char *text, unsigned char color[4], void *vertex_buffer, int vbuf_size);

// --- Implementierungen ---
static void stb_easy_font_spacing(float spacing) { stb_easy_font_spacing_val = spacing; }

static int stb_easy_font_print(float x, float y, char *text, unsigned char color[4], void *vertex_buffer, int vbuf_size)
{
    char *vbuf = (char *)vertex_buffer;
    float start_x = x;
    int offset = 0;
    stb_easy_font_color c = {255,255,255,255};
    if (color) { c.c[0]=color[0]; c.c[1]=color[1]; c.c[2]=color[2]; c.c[3]=color[3]; }
    while (*text && offset < vbuf_size) {
        if (*text == '\n') { y += 12; x = start_x; }
        else {
            unsigned char adv = stb_easy_font_charinfo[*text-32].advance;
            float y_ch = (adv & 16) ? y+1 : y;
            int h_seg = stb_easy_font_charinfo[*text-32].h_seg;
            int v_seg = stb_easy_font_charinfo[*text-32].v_seg;
            int num_h = stb_easy_font_charinfo[*text-32+1].h_seg - h_seg;
            int num_v = stb_easy_font_charinfo[*text-32+1].v_seg - v_seg;
            offset = stb_easy_font_draw_segs(x, y_ch, &stb_easy_font_hseg[h_seg], num_h, 0, c, vbuf, vbuf_size, offset);
            offset = stb_easy_font_draw_segs(x, y_ch, &stb_easy_font_vseg[v_seg], num_v, 1, c, vbuf, vbuf_size, offset);
            x += adv & 15;
            x += stb_easy_font_spacing_val;
        }
        ++text;
    }
    return (unsigned)offset / 64;
}

static int stb_easy_font_width(char *text)
{
    float len = 0, max_len = 0;
    while (*text) {
        if (*text == '\n') { if (len > max_len) max_len = len; len = 0; }
        else { len += stb_easy_font_charinfo[*text-32].advance & 15; len += stb_easy_font_spacing_val; }
        ++text;
    }
    if (len > max_len) max_len = len;
    return (int)ceil(max_len);
}

static int stb_easy_font_height(char *text)
{
    float y = 0; int nonempty = 0;
    while (*text) { if (*text == '\n') { y += 12; nonempty = 0; } else { nonempty = 1; } ++text; }
    return (int)ceil(y + (nonempty ? 12 : 0));
}

#endif // INCLUDE_STB_EASY_FONT_H
/*
This software is available under 2 licenses -- choose whichever you prefer.
ALTERNATIVE A - MIT License
Copyright (c) 2017 Sean Barrett
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
ALTERNATIVE B - Public Domain (www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain. We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors. We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
