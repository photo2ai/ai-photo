package com.photo2ai.aiphoto.core

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

/** 用仓库 models/ 里的真模型跑（桌面版 ONNX Runtime），核对输入输出约定和分块拼接 */
class LocalAiTest {
    private val models = File("../models")

    private fun gradient(w: Int, h: Int) = IntArray(w * h) {
        val x = it % w
        val y = it / w
        Pixels.rgb(-1, x * 255 / w, y * 255 / h, 128)
    }

    @Test
    fun miganFillsOnlyTheMaskedArea() {
        val w = 200
        val h = 160
        val clean = gradient(w, h)
        val dirty = clean.copyOf()
        val mask = FloatArray(w * h)
        for (y in 60 until 100) for (x in 80 until 120) {
            dirty[y * w + x] = 0xFFFF00FF.toInt() // 洋红色方块，要去掉
            mask[y * w + x] = 1f
        }
        val out = LocalAi.inpaint(File(models, LocalModels.MIGAN.file), dirty, w, h, mask)
        for (i in out.indices) if (mask[i] == 0f) assertEquals("蒙版外不应改动", dirty[i], out[i])
        var err = 0.0
        var n = 0
        for (y in 60 until 100) for (x in 80 until 120) {
            val i = y * w + x
            err += abs(Pixels.luma(out[i]) - Pixels.luma(clean[i])); n++
            assertTrue("洋红色没去掉", (out[i] shr 16 and 0xff) - (out[i] shr 8 and 0xff) < 120)
        }
        assertTrue("平均误差 ${err / n}", err / n < 25)
    }

    @Test
    fun esrganTilesMatchSinglePass() = runBlocking {
        val w = 70
        val h = 50
        val px = gradient(w, h)
        for (i in px.indices) if ((i % w) / 7 % 2 == (i / w) / 7 % 2) px[i] = px[i] xor 0x404040 // 加点纹理
        val model = File(models, LocalModels.ESRGAN.file)
        val whole = LocalAi.upscale4x(model, px, w, h, tile = 512) {}
        val tiled = LocalAi.upscale4x(model, px, w, h, tile = 24) {}
        assertEquals(w * 4 * h * 4, whole.size)
        var maxDiff = 0
        for (i in whole.indices) maxDiff = maxOf(maxDiff, abs(Pixels.luma(whole[i]) - Pixels.luma(tiled[i])))
        assertTrue("分块拼接和整张算的差别 $maxDiff", maxDiff <= 12)
    }
}
