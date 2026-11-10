package com.photo2ai.aiphoto.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MattingTest {
    @Test
    fun inputSizeFollowsModnetRule() {
        // 和电脑上 Python 跑的一样：1600×2400 → 512×768；长边 2048 的预览图（1365×2048）也一样
        assertEquals(512 to 768, Matting.inputSize(1600, 2400))
        assertEquals(512 to 768, Matting.inputSize(1365, 2048))
        assertEquals(768 to 512, Matting.inputSize(2400, 1600))
        // 本来就在 512 附近（一边大于、一边小于）的不缩，只取 32 的倍数
        assertEquals(384 to 608, Matting.inputSize(400, 620))
    }

    /**
     * 红色的人站在蓝色背景前，边缘有一条 40 像素宽的半透明带：估出来的前景色应该几乎不带蓝色。
     * 图要接近实际尺寸（模糊半径按图的大小缩放），电脑上同样的算法：蓝色 107 → 14
     */
    @Test
    fun foregroundRemovesBackgroundFromEdges() {
        val w = 1024
        val h = 40
        val alpha = FloatArray(w * h) { i -> ((120 - i % w) / 40f).coerceIn(0f, 1f) } // x 80..120 之间从 1 过渡到 0
        val px = IntArray(w * h) { i ->
            val a = alpha[i]
            Pixels.rgb(-1, Pixels.clamp255(220 * a), 0, Pixels.clamp255(220 * (1 - a)))
        }
        val fg = Matting.foreground(px, w, h, alpha)
        var before = 0f
        var after = 0f
        var n = 0
        for (y in 0 until h) for (x in 85 until 115) {
            val i = y * w + x
            before += (px[i] and 0xff).toFloat()
            after += (fg[i] and 0xff).toFloat()
            n++
        }
        assertTrue("边缘的蓝色分量：估计前 ${before / n}，估计后 ${after / n}", after / n < before / n * 0.25f)
        // 不透明的地方不变
        for (y in 0 until h) for (x in 0 until 60) assertEquals(px[y * w + x], fg[y * w + x])
    }
}
