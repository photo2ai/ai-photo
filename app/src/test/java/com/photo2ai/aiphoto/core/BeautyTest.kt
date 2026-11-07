package com.photo2ai.aiphoto.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/** 用画出来的假脸测美颜：肤色椭圆 + 噪点斑点、两只黑眼睛，网格点只给美颜用到的轮廓 */
class BeautyTest {
    private val skin = 0xFFE0AC8C.toInt()
    private val bg = 0xFF406080.toInt()

    private class Fake(val px: IntArray, val w: Int, val h: Int, val geom: Beauty.Geom)

    /** 脸中心 (cx, cy)、半宽 rx、半高 ry；网格点：0..35 外轮廓，36..51 左眼，52..67 右眼 */
    private fun fake(w: Int = 400, h: Int = 400, faces: List<FloatArray> = listOf(floatArrayOf(200f, 200f, 80f, 105f))): Fake {
        val rnd = Random(1)
        val px = IntArray(w * h) { bg }
        val geoms = ArrayList<Beauty.Geom>()
        for (f in faces) {
            val (cx, cy, rx, ry) = f.toList()
            val pts = FloatArray(68 * 2)
            for (i in 0 until 36) {
                val a = 2 * PI * i / 36 - PI / 2
                pts[2 * i] = (cx + rx * cos(a)).toFloat(); pts[2 * i + 1] = (cy + ry * sin(a)).toFloat()
            }
            val eyeR = rx * 0.18f
            for ((e, ex) in listOf(cx - rx * 0.4f, cx + rx * 0.4f).withIndex()) for (i in 0 until 16) {
                val a = 2 * PI * i / 16
                val k = 36 + e * 16 + i
                pts[2 * k] = (ex + eyeR * cos(a)).toFloat(); pts[2 * k + 1] = (cy - ry * 0.2f + eyeR * 0.5f * sin(a)).toFloat()
            }
            for (y in 0 until h) for (x in 0 until w) {
                val dx = (x + 0.5f - cx) / rx; val dy = (y + 0.5f - cy) / ry
                if (dx * dx + dy * dy > 1f) continue
                // 皮肤上的小起伏（毛孔、斑点）
                val n = rnd.nextInt(-14, 15)
                val r = (skin shr 16 and 0xff) + n; val g = (skin shr 8 and 0xff) + n; val b = (skin and 0xff) + n
                px[y * w + x] = Pixels.rgb(skin, r, g, b)
            }
            for (e in 0..1) {
                val ex = cx + (if (e == 0) -1 else 1) * rx * 0.4f
                val ey = cy - ry * 0.2f
                for (y in 0 until h) for (x in 0 until w) {
                    val dx = (x + 0.5f - ex) / eyeR; val dy = (y + 0.5f - ey) / (eyeR * 0.5f)
                    if (dx * dx + dy * dy <= 1f) px[y * w + x] = 0xFF101010.toInt()
                }
            }
            val oval = IntArray(36) { it }
            val eyes = listOf(IntArray(16) { 36 + it }, IntArray(16) { 52 + it })
            geoms += Beauty.Geom(pts, oval, eyes, eyes)
        }
        return Fake(px, w, h, geoms[0]).also { extra = geoms }
    }

    private var extra: List<Beauty.Geom> = emptyList()

    private fun prep(f: Fake) = extra.map { Beauty.prepare(f.px, f.w, f.h, it)!! }

    private fun variance(px: IntArray, w: Int, x0: Int, y0: Int, x1: Int, y1: Int): Double {
        val v = ArrayList<Int>()
        for (y in y0 until y1) for (x in x0 until x1) v += Pixels.luma(px[y * w + x])
        val m = v.average()
        return v.sumOf { (it - m) * (it - m) } / v.size
    }

    private fun meanLuma(px: IntArray, w: Int, x0: Int, y0: Int, x1: Int, y1: Int): Double {
        var s = 0.0; var n = 0
        for (y in y0 until y1) for (x in x0 until x1) { s += Pixels.luma(px[y * w + x]); n++ }
        return s / n
    }

    private fun dark(px: IntArray) = px.count { Pixels.luma(it) < 60 }

    /** 某一行上肤色像素的宽度 */
    private fun skinSpan(px: IntArray, w: Int, y: Int): Int {
        var n = 0
        for (x in 0 until w) {
            val c = px[y * w + x]
            if ((c shr 16 and 0xff) > 150 && (c and 0xff) < 180) n++
        }
        return n
    }

    @Test
    fun zeroParamsChangeNothing() {
        val f = fake()
        val out = Beauty.apply(f.px, f.w, prep(f), Beauty.Params())
        assertTrue(out.contentEquals(f.px))
    }

    @Test
    fun smoothingFlattensSkinOnly() {
        val f = fake()
        val out = Beauty.apply(f.px, f.w, prep(f), Beauty.Params(smooth = 1f))
        // 脸颊（眼睛下面）的起伏明显变小
        val before = variance(f.px, f.w, 150, 230, 250, 270)
        val after = variance(out, f.w, 150, 230, 250, 270)
        assertTrue("before=$before after=$after", after < before * 0.4)
        // 远离脸的背景一个像素都不动；眼睛没被磨淡
        for (y in 0 until 40) for (x in 0 until f.w) assertEquals(f.px[y * f.w + x], out[y * f.w + x])
        assertEquals(dark(f.px).toDouble(), dark(out).toDouble(), dark(f.px) * 0.05)
    }

    @Test
    fun whiteningBrightensFace() {
        val f = fake()
        val out = Beauty.apply(f.px, f.w, prep(f), Beauty.Params(whiten = 1f))
        val before = meanLuma(f.px, f.w, 160, 220, 240, 280)
        val after = meanLuma(out, f.w, 160, 220, 240, 280)
        assertTrue("before=$before after=$after", after > before + 10)
        // 背景（蓝灰色，不像肤色）不变亮
        assertEquals(meanLuma(f.px, f.w, 0, 0, 30, 400), meanLuma(out, f.w, 0, 0, 30, 400), 0.5)
    }

    @Test
    fun slimNarrowsJawNotForehead() {
        val f = fake()
        val out = Beauty.apply(f.px, f.w, prep(f), Beauty.Params(slim = 1f))
        // 下颌那一行（眼睛到下巴的六成处）变窄，额头不动
        val jawY = 200 + ((0.2f * 105 + 105) * 0.6f - 0.2f * 105).toInt()
        assertTrue(skinSpan(out, f.w, jawY) < skinSpan(f.px, f.w, jawY) - 4)
        assertEquals(skinSpan(f.px, f.w, 120), skinSpan(out, f.w, 120))
    }

    @Test
    fun eyesGetBigger() {
        val f = fake()
        val out = Beauty.apply(f.px, f.w, prep(f), Beauty.Params(eyes = 1f))
        assertTrue("${dark(f.px)} -> ${dark(out)}", dark(out) > dark(f.px) * 1.2)
    }

    @Test
    fun overlappingFacesBothProcessed() {
        // 两张脸挨得近，裁出的块互相重叠：后处理的脸不能把前一张脸的结果盖回原样
        val f = fake(w = 520, h = 300, faces = listOf(floatArrayOf(150f, 150f, 70f, 90f), floatArrayOf(370f, 150f, 70f, 90f)))
        val out = Beauty.apply(f.px, f.w, prep(f), Beauty.Params(smooth = 1f))
        for (cx in listOf(150, 370)) {
            val before = variance(f.px, f.w, cx - 40, 170, cx + 40, 200)
            val after = variance(out, f.w, cx - 40, 170, cx + 40, 200)
            assertTrue("face at $cx: before=$before after=$after", after < before * 0.4)
        }
    }

    @Test
    fun tinyFaceSkipped() {
        val f = fake(faces = listOf(floatArrayOf(200f, 200f, 8f, 10f)))
        assertEquals(null, Beauty.prepare(f.px, f.w, f.h, extra[0]))
    }
}
