package com.photo2ai.aiphoto.core

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalPartsTest {
    @Test
    fun chwRoundTripKeepsAlpha() {
        val px = intArrayOf(0x80112233.toInt(), 0xFFAABBCC.toInt(), 0x00010203)
        val b = Tensors.toChwU8(px)
        assertEquals(0x11, b[0].toInt() and 0xff)
        assertEquals(0xBB, b[3 + 1].toInt() and 0xff)
        assertTrue(Tensors.fromChwU8(b, px).contentEquals(px))
        assertEquals(listOf(0, 255), Tensors.miganMask(floatArrayOf(1f, 0f)).map { it.toInt() and 0xff })
    }

    @Test
    fun tilesCoverEveryPixelOnce() {
        for ((w, h) in listOf(100 to 70, 192 to 192, 5 to 400)) {
            val hits = IntArray(w * h)
            for (t in Tensors.tiles(w, h, 48, 8)) {
                assertTrue(t.x <= t.cx && t.y <= t.cy && t.x + t.w >= t.cx + t.cw && t.y + t.h >= t.cy + t.ch)
                assertTrue(t.x >= 0 && t.y >= 0 && t.x + t.w <= w && t.y + t.h <= h)
                for (y in t.cy until t.cy + t.ch) for (x in t.cx until t.cx + t.cw) hits[y * w + x]++
            }
            assertTrue("${w}x$h", hits.all { it == 1 })
        }
    }

    @Test
    fun putScaledTileWritesCenterOnly() {
        val w = 4
        val h = 2
        val src = IntArray(w * h) { 0xFF000000.toInt() }
        val out = IntArray(w * 2 * h * 2)
        val t = Tensors.Tile(1, 0, 3, 2, 2, 0, 2, 2)
        val data = FloatArray(3 * 6 * 4) { 1f } // 整块白色
        Tensors.putScaledTile(out, w * 2, t, data, 2, src, w)
        for (y in 0 until 4) for (x in 0 until 8) {
            val c = out[y * 8 + x]
            if (x >= 4) assertEquals(0xFFFFFFFF.toInt(), c) else assertEquals(0, c)
        }
    }

    @Test
    fun seedreamSizeKeepsAspectWithinLimits() {
        for ((w, h) in listOf(1600 to 2400, 4000 to 3000, 1000 to 1000, 3000 to 100)) {
            val (ow, oh) = CloudProtocol.seedreamSize(w, h).split("x").map { it.toInt() }
            assertTrue(ow % 8 == 0 && oh % 8 == 0)
            assertTrue("$ow x $oh", ow.toLong() * oh in 1280L * 720..4096L * 4096 && ow <= 4096 && oh <= 4096)
            val a = (w.toDouble() / h).coerceIn(1.0 / 16, 16.0)
            assertTrue("aspect", kotlin.math.abs(ow.toDouble() / oh - a) / a < 0.03)
        }
    }

    @Test
    fun seedreamBodyInlinesImages() {
        val one = JSONObject(CloudProtocol.seedreamBody("m", "p", listOf(byteArrayOf(1)), "1024x1024"))
        assertTrue(one.getString("image").startsWith("data:image/jpeg;base64,"))
        assertEquals("b64_json", one.getString("response_format"))
        assertEquals(false, one.getBoolean("watermark"))
        val two = JSONObject(CloudProtocol.seedreamBody("m", "p", listOf(byteArrayOf(1), byteArrayOf(2)), "1024x1024"))
        assertEquals(2, two.getJSONArray("image").length())
        assertEquals("https://ark.cn-beijing.volces.com/api/v3/images/generations", CloudProtocol.seedreamUrl("https://ark.cn-beijing.volces.com/api/v3/"))
    }

    @Test
    fun faceWarpIdentityCopiesTriangle() {
        val w = 40
        val h = 40
        val src = IntArray(w * h) { Pixels.rgb(-1, it % w * 6, it / w * 6, 99) }
        val pts = floatArrayOf(5f, 5f, 35f, 5f, 5f, 35f)
        val (out, cover) = FaceSwap.warp(src, w, h, pts, w, h, pts, intArrayOf(0, 1, 2))
        assertEquals(1f, cover[10 * w + 10])
        assertEquals(0f, cover[30 * w + 30])
        val i = 10 * w + 12
        assertTrue(kotlin.math.abs(Pixels.luma(out[i]) - Pixels.luma(src[i])) <= 2)
    }

    @Test
    fun faceMaskIsSoftInsidePolygon() {
        val w = 100
        val h = 100
        val poly = floatArrayOf(20f, 20f, 80f, 20f, 80f, 80f, 20f, 80f)
        val m = FaceSwap.faceMask(poly, w, h)
        assertEquals(1f, m[50 * w + 50], 0.01f)
        assertEquals(0f, m[5 * w + 5], 0.01f)
        assertTrue(m[50 * w + 21] < 0.5f) // 边缘羽化
    }

    @Test
    fun matchColorMovesMeanToTarget() {
        val n = 400
        val src = IntArray(n) { Pixels.rgb(-1, 200, 150, 120) }
        val target = IntArray(n) { Pixels.rgb(-1, 120, 80, 60) }
        val out = FaceSwap.matchColor(src, target, FloatArray(n) { 1f })
        assertEquals(120f, (out[0] shr 16 and 0xff).toFloat(), 2f)
        assertEquals(60f, (out[0] and 0xff).toFloat(), 2f)
    }

    @Test
    fun matchColorDesaturatesToSepiaTarget() {
        // 彩色的脸贴到几乎没有颜色的老照片上：结果也应该几乎没有颜色
        val n = 400
        val src = IntArray(n) { if (it % 2 == 0) Pixels.rgb(-1, 220, 140, 110) else Pixels.rgb(-1, 60, 70, 160) }
        val target = IntArray(n) { val v = 100 + it % 50; Pixels.rgb(-1, v + 12, v + 6, v) }
        val out = FaceSwap.matchColor(src, target, FloatArray(n) { 1f })
        for (c in out) {
            val r = c shr 16 and 0xff
            val b = c and 0xff
            assertTrue("还有明显的颜色：r=$r b=$b", kotlin.math.abs(r - b) < 30)
        }
    }

    /** 启动时会删掉被取代的旧模型文件：它们绝不能是现在还在用的文件名 */
    @Test
    fun replacedModelFilesAreNotInUse() {
        val all = listOf(LocalModels.MIGAN, LocalModels.ESRGAN, LocalModels.ARCFACE, LocalModels.INSWAPPER)
        val inUse = all.map { it.file }.toSet()
        for (m in all) m.replaces?.let { org.junit.Assert.assertFalse("${m.file} 取代的 $it 还在用", it in inUse) }
    }
}
