package com.photo2ai.aiphoto.core

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * 一键增强（本机运行）：色阶拉伸 + 白平衡、亮度校正、局部对比（清晰度）、自然饱和度、轻微锐化。
 */
object Enhance {
    fun auto(px: IntArray, w: Int, h: Int): IntArray {
        // 1. 采样直方图（透明像素不算）
        val step = max(1, px.size / 250_000)
        val hr = IntArray(256)
        val hg = IntArray(256)
        val hb = IntArray(256)
        var n = 0
        var i = 0
        while (i < px.size) {
            val c = px[i]
            if ((c ushr 24) > 16) {
                hr[c shr 16 and 0xff]++; hg[c shr 8 and 0xff]++; hb[c and 0xff]++
                n++
            }
            i += step
        }
        if (n == 0) return px.copyOf()
        val lo = intArrayOf(percentile(hr, n, 0.005f), percentile(hg, n, 0.005f), percentile(hb, n, 0.005f))
        val hi = intArrayOf(percentile(hr, n, 0.995f), percentile(hg, n, 0.995f), percentile(hb, n, 0.995f))
        val loMean = lo.average().toFloat()
        val hiMean = hi.average().toFloat()

        // 2. 白平衡：每个通道的黑白点往三通道的平均靠 60%（全靠过去会把夕阳的暖色也洗掉）；
        //    色阶：再按共同的黑白点拉伸，最多拉伸 2.5 倍
        val range = max(hiMean - loMean, 255f / 2.5f)
        val levels = Array(3) { ch ->
            val loT = lo[ch] + (loMean - lo[ch]) * WB
            val hiT = hi[ch] + (hiMean - hi[ch]) * WB
            val gain = (hiT - loT) / max(hi[ch] - lo[ch], 1).toFloat()
            FloatArray(256) { v -> ((loT + (v - lo[ch]) * gain - loMean) / range).coerceIn(0f, 1f) }
        }

        // 3. 亮度：偏暗的往 0.48 提亮；正常和偏亮的不压暗（压暗会让肤色发浓发脏），只有明显过曝（> 0.68）才往 0.6 轻压
        var sum = 0.0
        var cnt = 0
        i = 0
        while (i < px.size) {
            val c = px[i]
            if ((c ushr 24) > 16) {
                sum += 0.299 * levels[0][c shr 16 and 0xff] + 0.587 * levels[1][c shr 8 and 0xff] + 0.114 * levels[2][c and 0xff]
                cnt++
            }
            i += step
        }
        val mean = (sum / max(cnt, 1)).toFloat().coerceIn(0.02f, 0.98f)
        val gamma = when {
            mean < 0.48f -> (ln(0.48f) / ln(mean)).coerceIn(0.55f, 1f)
            mean > 0.68f -> (ln(0.6f) / ln(mean)).coerceIn(1f, 1.3f)
            else -> 1f
        }
        val luts = Array(3) { ch -> IntArray(256) { v -> Pixels.clamp255(levels[ch][v].pow(gamma) * 255f) } }

        val out = IntArray(px.size)
        for (k in px.indices) {
            val c = px[k]
            out[k] = Pixels.rgb(c, luts[0][c shr 16 and 0xff], luts[1][c shr 8 and 0xff], luts[2][c and 0xff])
        }

        // 4. 清晰度：亮度减去中等半径模糊得到局部细节，加回 25%。半径不能大：大到和脸差不多时，
        //    被白衣服、浅背景围着的脸会被整体压暗
        val lum = FloatArray(out.size) { Pixels.luma(out[it]).toFloat() }
        val blur = Pixels.blurF(lum, w, h, max(2, max(w, h) / 250), 2)
        // 5. 自然饱和度：越不饱和的颜色加得越多；肤色（R > G > B 的橙色调）不加，色阶拉伸已经让它变浓了
        for (k in out.indices) {
            val c = out[k]
            val d = (lum[k] - blur[k]) * 0.25f
            var r = (c shr 16 and 0xff) + d
            var g = (c shr 8 and 0xff) + d
            var b = (c and 0xff) + d
            val mx = max(r, max(g, b))
            val mn = min(r, min(g, b))
            val s = ((mx - mn) / 255f).coerceIn(0f, 1f)
            val skin = r > g && g > b && (r - g) < 0.8f * (r - b) && r > 60
            val boost = if (skin) 1f else 1f + 0.15f * (1f - s)
            val l = 0.299f * r + 0.587f * g + 0.114f * b
            r = l + (r - l) * boost
            g = l + (g - l) * boost
            b = l + (b - l) * boost
            out[k] = Pixels.rgb(c, Pixels.clamp255(r), Pixels.clamp255(g), Pixels.clamp255(b))
        }

        // 6. 轻微锐化
        return Pixels.sharpen(out, w, h, 0.25f, max(1, max(w, h) / 2000))
    }

    private const val WB = 0.6f

    private fun percentile(hist: IntArray, n: Int, p: Float): Int {
        val target = (n * p).toLong()
        var acc = 0L
        for (v in 0..255) {
            acc += hist[v]
            if (acc > target) return v
        }
        return 255
    }
}
