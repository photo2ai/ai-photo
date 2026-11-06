package com.photo2ai.aiphoto.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class PixelOpsTest {
    private fun solid(n: Int, c: Int) = IntArray(n) { c }
    private fun r(c: Int) = c shr 16 and 0xff
    private fun g(c: Int) = c shr 8 and 0xff
    private fun b(c: Int) = c and 0xff

    @Test
    fun blurKeepsFlatImage() {
        val px = solid(40 * 30, 0xFF336699.toInt())
        val out = Pixels.boxBlur(px, 40, 30, 4)
        assertTrue(out.all { it == 0xFF336699.toInt() })
    }

    @Test
    fun adjustIdentityAndDirections() {
        val w = 16
        val h = 16
        val px = IntArray(w * h) { val v = (it * 255 / (w * h)); (0xFF shl 24) or (v shl 16) or ((v / 2) shl 8) or (255 - v) }
        assertTrue(Adjust().apply(px, w, h).contentEquals(px))
        val brighter = Adjust(brightness = 0.5f).apply(px, w, h)
        assertTrue(brighter.sumOf { Pixels.luma(it) } > px.sumOf { Pixels.luma(it) })
        val mono = Adjust(saturation = -1f).apply(px, w, h)
        assertTrue(mono.all { abs(r(it) - g(it)) <= 1 && abs(g(it) - b(it)) <= 1 })
        val warm = Adjust(warmth = 1f).apply(px, w, h)
        assertTrue(warm.sumOf { r(it) - b(it) } > px.sumOf { r(it) - b(it) })
    }

    @Test
    fun toneLutIsMonotonic() {
        for (s in listOf(-1f, 1f)) {
            val lut = Adjust(shadows = s, highlights = -s, contrast = s * 0.5f, fade = 0.5f).toneLut()
            for (i in 1..255) assertTrue("shadows=$s at $i", lut[i] >= lut[i - 1])
        }
    }

    @Test
    fun enhanceStretchesDarkFlatImage() {
        val w = 64
        val h = 48
        // 偏暗、低对比
        val px = IntArray(w * h) { val v = 40 + (it % w) * 40 / w; (0xFF shl 24) or (v shl 16) or (v shl 8) or v }
        val out = Enhance.auto(px, w, h)
        val inRange = px.maxOf { Pixels.luma(it) } - px.minOf { Pixels.luma(it) }
        val outRange = out.maxOf { Pixels.luma(it) } - out.minOf { Pixels.luma(it) }
        assertTrue("range $inRange -> $outRange", outRange > inRange * 2)
        val meanIn = px.map { Pixels.luma(it) }.average()
        val meanOut = out.map { Pixels.luma(it) }.average()
        assertTrue("mean $meanIn -> $meanOut", meanOut > meanIn + 30)
    }

    @Test
    fun enhanceReducesColorCast() {
        val w = 64
        val h = 48
        // 正常亮度范围，整体偏蓝
        val px = IntArray(w * h) { val v = 30 + (it % w) * 160 / w; (0xFF shl 24) or (v shl 16) or (v shl 8) or (v + 30) }
        val out = Enhance.auto(px, w, h)
        // 偏色相对整体反差算：拉伸会把所有差值一起放大
        fun cast(a: IntArray) = a.map { b(it) - r(it) }.average() / (a.maxOf { Pixels.luma(it) } - a.minOf { Pixels.luma(it) })
        assertTrue("cast ${cast(px)} -> ${cast(out)}", cast(out) < cast(px) * 0.6)
    }

    @Test
    fun enhanceKeepsAlpha() {
        val px = IntArray(100) { if (it < 50) 0x00000000 else 0xFF808080.toInt() }
        val out = Enhance.auto(px, 10, 10)
        for (i in px.indices) assertEquals(px[i] ushr 24, out[i] ushr 24)
    }

    @Test
    fun inpaintFillsHoleFromSurroundings() {
        val w = 50
        val h = 40
        // 横向渐变，中间画一个黑块再涂掉
        val clean = IntArray(w * h) { val x = it % w; val v = x * 5; (0xFF shl 24) or (v shl 16) or (v shl 8) or v }
        val dirty = clean.copyOf()
        val mask = FloatArray(w * h)
        for (y in 15 until 25) for (x in 20 until 30) {
            dirty[y * w + x] = 0xFF000000.toInt()
            mask[y * w + x] = 1f
        }
        val out = Inpaint.fill(dirty, w, h, mask)
        var err = 0
        for (y in 15 until 25) for (x in 20 until 30) err = maxOf(err, abs(Pixels.luma(out[y * w + x]) - Pixels.luma(clean[y * w + x])))
        assertTrue("max error $err", err < 30)
        // 蒙版外不动
        for (i in clean.indices) if (mask[i] == 0f) assertEquals(dirty[i], out[i])
    }

    @Test
    fun inpaintWithoutMaskIsCopy() {
        val px = IntArray(16) { 0xFF112233.toInt() + it }
        assertTrue(Inpaint.fill(px, 4, 4, FloatArray(16)).contentEquals(px))
    }

    @Test
    fun guidedFilterSnapsToEdge() {
        val w = 40
        val h = 20
        // 图像左黑右白，粗糙蒙版在边缘处是一段斜坡
        val img = IntArray(w * h) { if (it % w < 20) 0xFF000000.toInt() else 0xFFFFFFFF.toInt() }
        val coarse = FloatArray(w * h) { ((it % w - 14) / 12f).coerceIn(0f, 1f) }
        val m = Cutout.refine(img, coarse, w, h)
        val row = 10 * w
        assertTrue("left ${m[row + 16]}", m[row + 16] < 0.25f)
        assertTrue("right ${m[row + 23]}", m[row + 23] > 0.75f)
    }

    @Test
    fun cutoutTransparentKeepsColor() {
        val px = IntArray(4) { 0xFFAA5500.toInt() }
        val out = Cutout.transparent(px, floatArrayOf(1f, 0.5f, 0f, 0.25f))
        assertEquals(255, out[0] ushr 24)
        assertEquals(128, out[1] ushr 24)
        assertEquals(0, out[2] ushr 24)
        assertTrue(out.all { it and 0xffffff == 0xAA5500 })
    }

    @Test
    fun resizeMaskBilinear() {
        val m = Pixels.resizeF(floatArrayOf(0f, 1f), 2, 1, 4, 1)
        assertEquals(0f, m[0], 1e-4f)
        assertEquals(1f, m[3], 1e-4f)
        assertTrue(m[1] > 0f && m[2] < 1f && m[1] < m[2])
    }
}
