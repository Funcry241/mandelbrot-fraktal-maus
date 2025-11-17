;///// Otter: Scalar Mandelbrot-Iteration als erste ASM-Basis für OtterDream.
;///// Schneefuchs: Saubere Win64-ABI, keine Nonvolatile-Register verletzt.
;///// Maus: Startpunkt für spätere Zeilen-/Tile-Renderer in ASM.
;///// Datei: src/asm/mandelbrot_iter.asm

option casemap:none

; Extern sichtbare Funktion:
; int mandelbrotIter_asm(double x0, double y0, int maxIters);
;
; Win64-Calling-Convention (MSVC/Clang-cl):
;   x0       -> XMM0  (double)
;   y0       -> XMM1  (double)
;   maxIters -> R8D   (int, zero-extended)
; Rückgabewert:
;   EAX (int) = Anzahl Iterationen bis Escape oder maxIters

PUBLIC mandelbrotIter_asm

.const
ALIGN 8
const_four REAL8 4.0           ; Escape-Radius² = 4.0

.code

mandelbrotIter_asm PROC

    ; Stack-Frame: wir reservieren 32 Bytes für x und y (2 * 8) + Alignment.
    ; [rsp]     : double x
    ; [rsp+8]   : double y
    ; Rest ungenutzt, aber 16-Byte-Alignment bleibt gewahrt.

    sub     rsp, 32

    ; x = 0.0, y = 0.0
    pxor    xmm2, xmm2          ; xmm2 = 0.0
    movsd   QWORD PTR [rsp], xmm2       ; x
    movsd   QWORD PTR [rsp+8], xmm2     ; y

    xor     eax, eax            ; i = 0

LoopBegin:

    ; if (i >= maxIters) goto Done;
    cmp     eax, r8d
    jge     Done

    ; x_alt und y_alt laden
    movsd   xmm2, QWORD PTR [rsp]       ; x
    movsd   xmm3, QWORD PTR [rsp+8]     ; y

    ; x2 = x*x
    movsd   xmm4, xmm2
    mulsd   xmm4, xmm2                  ; xmm4 = x^2

    ; y2 = y*y
    movsd   xmm5, xmm3
    mulsd   xmm5, xmm3                  ; xmm5 = y^2

    ; newx = x2 - y2 + x0
    subsd   xmm4, xmm5                  ; x^2 - y^2
    addsd   xmm4, xmm0                  ; + x0  -> newx in xmm4

    ; newy = 2*x*y + y0
    movsd   xmm5, xmm2                  ; xmm5 = x
    mulsd   xmm5, xmm3                  ; x*y
    addsd   xmm5, xmm5                  ; 2*x*y
    addsd   xmm5, xmm1                  ; + y0  -> newy in xmm5

    ; x = newx; y = newy;
    movsd   QWORD PTR [rsp],   xmm4     ; x = newx
    movsd   QWORD PTR [rsp+8], xmm5     ; y = newy

    ; i++ (Anzahl ausgeführter Iterationen)
    inc     eax

    ; Escape-Test: |z|² = x*x + y*y > 4.0 ?
    movsd   xmm2, QWORD PTR [rsp]       ; x
    mulsd   xmm2, xmm2                  ; x^2

    movsd   xmm3, QWORD PTR [rsp+8]     ; y
    mulsd   xmm3, xmm3                  ; y^2

    addsd   xmm2, xmm3                  ; x^2 + y^2

    movsd   xmm4, QWORD PTR [const_four]
    ucomisd xmm2, xmm4
    ja      Escape                       ; wenn > 4.0 -> Escape

    jmp     LoopBegin

Escape:
Done:
    add     rsp, 32
    ret

mandelbrotIter_asm ENDP

END
