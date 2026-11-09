package com.photo2ai.aiphoto.core

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

class FaceSwapModelTest {
    @Test
    fun similarityRecoversRotationScaleShift() {
        // 模板放大 3 倍、转 20°、平移 (40, -10) 后，求出来的变换要能把它映射回模板
        val t = FaceAlign.ARCFACE_112
        val a = Math.toRadians(20.0)
        val src = FloatArray(10) { i ->
            val x = t[i - i % 2]
            val y = t[i - i % 2 + 1]
            if (i % 2 == 0) (3 * (x * Math.cos(a) - y * Math.sin(a)) + 40).toFloat() else (3 * (x * Math.sin(a) + y * Math.cos(a)) - 10).toFloat()
        }
        val m = FaceAlign.similarity(src, t)
        for (i in 0 until 5) {
            val p = FaceAlign.apply(m, src[2 * i], src[2 * i + 1])
            assertEquals(t[2 * i], p[0], 0.01f)
            assertEquals(t[2 * i + 1], p[1], 0.01f)
        }
        val back = FaceAlign.apply(FaceAlign.invert(m), t[0], t[1])
        assertEquals(src[0], back[0], 0.01f)
        assertEquals(src[1], back[1], 0.01f)
    }

    @Test
    fun fivePointsSortedLeftToRight() {
        val pts = FloatArray(400 * 2)
        fun put(i: Int, x: Float, y: Float) { pts[2 * i] = x; pts[2 * i + 1] = y }
        put(10, 80f, 50f); put(11, 90f, 50f)   // 右边那只眼
        put(20, 20f, 52f); put(21, 30f, 52f)   // 左边那只眼
        put(1, 55f, 70f)
        put(61, 75f, 90f); put(291, 35f, 90f)  // 嘴角故意反着给
        val f = FaceSwap.fivePoints(pts, intArrayOf(10, 11), intArrayOf(20, 21), 1, 61, 291)
        assertArrayEquals(floatArrayOf(25f, 52f, 85f, 50f, 55f, 70f, 35f, 90f, 75f, 90f), f, 0.001f)
    }

    @Test
    fun opennessComparison() {
        assertTrue(FaceSwap.similarOpenness(0.30f, 0.34f))   // 都睁眼
        assertTrue(FaceSwap.similarOpenness(0.02f, 0.06f))   // 都闭嘴
        assertFalse(FaceSwap.similarOpenness(0.08f, 0.35f))  // 眯眼和睁眼
        assertFalse(FaceSwap.similarOpenness(0.03f, 0.45f))  // 闭嘴和大笑
        assertFalse(FaceSwap.similarOpenness(0.18f, 0.32f))  // 笑得眯眼和正常睁眼（拉开会变两团黑）
    }

    /** 手工拼一个最小的 ONNX：graph 里两个 initializer，取最后一个 */
    private val man = File("/tmp/aiphoto-dl/portrait.jpg")
    private val woman = File("/tmp/aiphoto-dl/street.jpg")
    private val manFive = floatArrayOf(588f, 820f, 902f, 811f, 758f, 1008f, 636f, 1158f, 882f, 1154f)
    private val womanFive = floatArrayOf(666f, 708f, 902f, 710f, 780f, 828f, 678f, 930f, 882f, 930f)

    private fun read(f: File): Triple<IntArray, Int, Int> {
        val im = javax.imageio.ImageIO.read(f)
        return Triple(im.getRGB(0, 0, im.width, im.height, null, 0, im.width), im.width, im.height)
    }

    /** 女人的脸换到男人照片上，返回整张结果（男人照片的尺寸） */
    private fun swap(arc: File, swapper: File): IntArray {
        val (mp, mw, mh) = read(man)
        val (wp, ww, wh) = read(woman)
        val emb = NeuralSwap.embedding(arc, wp, ww, wh, womanFive)
        assertEquals(512, emb.size)
        return NeuralSwap.swap(swapper, mp, mw, mh, manFive, emb, null).also {
            LocalAi.release(arc)
            LocalAi.release(swapper)
        }
    }

    /** 原版 HyperSwap（半精度，FaceFusion 发布的那个文件）和人脸特征模型；电脑上有才跑，模型太大不放进仓库 */
    private val arc = File("/tmp/aiphoto-models/arcface_w600k_r50.onnx")
    private val hyper = File("/tmp/aiphoto-models/cand/hyperswap_1c_256.onnx")

    /** 真模型：参考脸换到目标脸上，脸部变了、远处不变 */
    @Test
    fun neuralSwapChangesFaceOnly() {
        assumeTrue(arc.isFile && hyper.isFile && man.isFile && woman.isFile)
        val out = swap(arc, hyper)
        val (mp, mw, _) = read(man)
        var changed = 0
        for (y in 850 until 1100) for (x in 600 until 900) if (abs(Pixels.luma(out[y * mw + x]) - Pixels.luma(mp[y * mw + x])) > 8) changed++
        assertTrue("脸部改动的像素 $changed", changed > 20000)
        for (y in 0 until 200) for (x in 0 until mw) assertEquals(mp[y * mw + x], out[y * mw + x])
    }

    /**
     * App 用的量化版（ArcFace INT8 + HyperSwap 只量化权重，scripts/ 下的脚本生成到 /tmp/aiphoto-models/q/，有才跑）：
     * 换出来的脸和原版一对的结果接近
     */
    @Test
    fun neuralSwapQuantizedMatchesOriginal() {
        val arc8 = File("/tmp/aiphoto-models/q/" + LocalModels.ARCFACE.file)
        val hyper8 = File("/tmp/aiphoto-models/q/" + LocalModels.HYPERSWAP.file)
        assumeTrue(arc.isFile && hyper.isFile && arc8.isFile && hyper8.isFile && man.isFile && woman.isFile)
        val a = swap(arc, hyper)
        val b = swap(arc8, hyper8)
        val (_, mw, _) = read(man)
        var diff = 0L
        var n = 0
        for (y in 850 until 1100) for (x in 600 until 900) {
            diff += abs(Pixels.luma(a[y * mw + x]) - Pixels.luma(b[y * mw + x]))
            n++
        }
        assertTrue("脸部平均亮度差 ${diff.toFloat() / n}", diff.toFloat() / n < 4f)
    }
}
