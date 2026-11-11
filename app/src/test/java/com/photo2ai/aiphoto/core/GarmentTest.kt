package com.photo2ai.aiphoto.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/** 本地换装的衣服拆分和蒙版工具：用画出来的几种衣服形状检查（真实衣服照片的效果在电脑原型上对比过） */
class GarmentTest {
    private val w = 300
    private val h = 400

    private fun blank() = BooleanArray(w * h)
    private fun rect(m: BooleanArray, x0: Int, y0: Int, x1: Int, y1: Int) {
        for (y in y0 until y1) for (x in x0 until x1) m[y * w + x] = true
    }
    private fun poly(m: BooleanArray, vararg p: Float) = Masks.fillConvex(w, h, p.toList().chunked(2).map { V(it[0], it[1]) }, m)

    /** T 恤：躯干 100..200 × 60..340，两只袖子从肩往外斜伸 */
    private fun tee(): BooleanArray {
        val m = blank()
        rect(m, 100, 60, 200, 340)
        poly(m, 100f, 60f, 100f, 150f, 50f, 170f, 30f, 120f)
        poly(m, 200f, 60f, 270f, 120f, 250f, 170f, 200f, 150f)
        return m
    }

    @Test
    fun teeSplitsTorsoAndOutwardSleeves() {
        val g = Garment.parse(tee(), w, h)
        assertEquals(setOf('L', 'R'), g.sleeves.keys)
        // 躯干边线在 100、200 附近，肩在顶上，腋下在袖子下沿附近，下摆在 340 附近
        assertTrue(abs(g.sl.x - 100) <= 3 && abs(g.sr.x - 199) <= 3)
        assertTrue(abs(g.sl.y - 60) <= 3)
        assertTrue(g.al.y in 140f..175f)
        assertTrue(abs(g.hem - 339) <= 2)
        // 躯干里没有袖子的像素，袖子朝外（左袖指向左下、右袖指向右下）
        assertFalse(g.torso[130 * w + 40])
        assertTrue(g.torso[200 * w + 150])
        assertTrue(g.sleeves['L']!!.axis.x < 0 && g.sleeves['R']!!.axis.x > 0)
    }

    @Test
    fun hangingLongSleevesAreFoundFromCuffsBelowHem() {
        // 开衫：躯干 100..200 × 40..300，两只长袖贴着身体垂下，袖口比下摆低
        val m = blank()
        rect(m, 100, 40, 200, 300)
        poly(m, 100f, 40f, 100f, 340f, 70f, 340f, 75f, 60f)
        poly(m, 200f, 40f, 225f, 60f, 230f, 340f, 200f, 340f)
        val g = Garment.parse(m, w, h)
        assertEquals(setOf('L', 'R'), g.sleeves.keys)
        assertTrue(abs(g.sl.x - 101) <= 3 && abs(g.sr.x - 198) <= 3)
        // 袖子朝下
        assertTrue(g.sleeves['L']!!.axis.y > 0.9f && g.sleeves['R']!!.axis.y > 0.9f)
        assertTrue(g.torso[200 * w + 150] && !g.torso[320 * w + 85])
    }

    @Test
    fun sideTieDoesNotWidenTorso() {
        // T 恤右边腰上多一条往外、往下垂的系带：躯干边线还是按左右对称的窄边算
        val m = tee()
        rect(m, 200, 230, 245, 380)
        val g = Garment.parse(m, w, h)
        assertTrue("右下摆 ${g.hr.x}", g.hr.x < 205)
        assertTrue(abs((g.cx - g.hl.x) - (g.hr.x - g.cx)) < 4)
        // 系带算躯干的一部分（跟着躯干变形），不算袖子
        assertTrue(g.torso[300 * w + 232] && g.torso[370 * w + 240])
    }

    @Test
    fun cutoutFindsGarmentOnPlainBackground() {
        val px = IntArray(w * h) { 0xFFEEEEEE.toInt() }
        val m = tee()
        for (i in px.indices) if (m[i]) px[i] = 0xFF202020.toInt()
        px[10 * w + 10] = 0xFF000000.toInt() // 零星噪点不算
        val cut = Garment.cutout(px, w, h)
        assertEquals(Masks.count(m).toFloat(), Masks.count(cut).toFloat(), Masks.count(m) * 0.02f)
        assertFalse(cut[10 * w + 10])
    }

    @Test
    fun cutoutUsesAlphaForTransparentPng() {
        // 透明背景上的黑色 T 恤：透明像素颜色也是黑的，按颜色分不开，要按透明度
        val m = tee()
        val px = IntArray(w * h) { if (m[it]) 0xFF101010.toInt() else 0x00000000 }
        val cut = Garment.cutout(px, w, h)
        assertEquals(Masks.count(m).toFloat(), Masks.count(cut).toFloat(), Masks.count(m) * 0.02f)
    }

    @Test(expected = Garment.Bad::class)
    fun cutoutRejectsPhotoWithoutGarment() {
        Garment.cutout(IntArray(w * h) { 0xFFEEEEEE.toInt() }, w, h)
    }

    @Test
    fun maskOps() {
        val m = blank()
        rect(m, 50, 50, 60, 60)
        assertEquals(100, Masks.count(m))
        assertEquals(14 * 14, Masks.count(Masks.dilate(m, w, h, 2)))
        assertEquals(6 * 6, Masks.count(Masks.erode(m, w, h, 2)))
        assertEquals(100, Masks.count(Masks.open(m, w, h, 2)))
        // 带洞的环填洞后是实心的
        val ring = blank()
        rect(ring, 10, 10, 40, 40)
        for (y in 20 until 30) for (x in 20 until 30) ring[y * w + x] = false
        assertEquals(900, Masks.count(Masks.fillHoles(ring, w, h)))
        // 只留最大块、去小块
        rect(m, 100, 100, 103, 103)
        assertEquals(100, Masks.count(Masks.largest(m, w, h)))
        assertEquals(100, Masks.count(Masks.dropSmall(m, w, h, 20)))
        // 凸包和胶囊
        val hull = Masks.convexHull(listOf(V(0f, 0f), V(10f, 0f), V(10f, 10f), V(0f, 10f), V(5f, 5f)))
        assertEquals(4, hull.size)
        val cap = Masks.capsule(w, h, listOf(V(100f, 100f), V(200f, 100f)), 5f)
        assertTrue(cap[100 * w + 150] && cap[104 * w + 150] && !cap[107 * w + 150] && cap[100 * w + 204] && !cap[100 * w + 206])
        assertEquals(50f, Masks.percentile(FloatArray(101) { it.toFloat() }, 50f), 1e-4f)
    }

    @Test
    fun labRoundTrip() {
        for (c in listOf(floatArrayOf(0f, 0f, 0f), floatArrayOf(1f, 1f, 1f), floatArrayOf(0.2f, 0.5f, 0.8f), floatArrayOf(0.9f, 0.1f, 0.3f))) {
            val lab = TryOn.Color.rgbToLab(c[0], c[1], c[2])
            val back = TryOn.Color.labToRgb(lab[0], lab[1], lab[2])
            for (k in 0..2) assertEquals(c[k], back[k], 2e-3f)
        }
        // 整张图的查表版和逐个算的一致
        val px = intArrayOf(0xFF000000.toInt(), 0xFFFFFFFF.toInt(), 0xFF3366CC.toInt(), 0xFFE61A4D.toInt(), 0xFF020202.toInt())
        val all = TryOn.Color.labOf(px)
        for (i in px.indices) {
            val c = px[i]
            val one = TryOn.Color.rgbToLab((c shr 16 and 0xff) / 255f, (c shr 8 and 0xff) / 255f, (c and 0xff) / 255f)
            for (k in 0..2) assertEquals(one[k], all[i * 3 + k], 1e-3f)
        }
        // 白色 L = 100，中性灰 a、b ≈ 0
        assertEquals(100f, TryOn.Color.rgbToLab(1f, 1f, 1f)[0], 0.1f)
        assertEquals(0f, TryOn.Color.rgbToLab(0.5f, 0.5f, 0.5f)[1], 0.1f)
    }
}
