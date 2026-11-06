package com.photo2ai.aiphoto.core

/** 人像蒙版的后处理和合成（纯计算，分割本身见 Segment） */
object Cutout {
    /**
     * 导向滤波（He et al.）：以原图亮度为引导把蒙版边缘贴到图像真实的边缘上。ML Kit 的模型输入只有 256x256，
     * 放大后的蒙版边缘是糊的，这一步让头发、肩膀的轮廓清楚很多。
     */
    fun guided(guide: FloatArray, p: FloatArray, w: Int, h: Int, r: Int, eps: Float): FloatArray {
        val n = guide.size
        val ip = FloatArray(n) { guide[it] * p[it] }
        val ii = FloatArray(n) { guide[it] * guide[it] }
        val meanI = Pixels.blurF(guide, w, h, r, 1)
        val meanP = Pixels.blurF(p, w, h, r, 1)
        val corrIp = Pixels.blurF(ip, w, h, r, 1)
        val corrI = Pixels.blurF(ii, w, h, r, 1)
        val a = FloatArray(n)
        val b = FloatArray(n)
        for (k in 0 until n) {
            val varI = corrI[k] - meanI[k] * meanI[k]
            val cov = corrIp[k] - meanI[k] * meanP[k]
            a[k] = cov / (varI + eps)
            b[k] = meanP[k] - a[k] * meanI[k]
        }
        val ma = Pixels.blurF(a, w, h, r, 1)
        val mb = Pixels.blurF(b, w, h, r, 1)
        return FloatArray(n) { (ma[it] * guide[it] + mb[it]).coerceIn(0f, 1f) }
    }

    /** 模型输出的置信度 → 干净的蒙版：先按图像边缘细化，再收紧过渡带 */
    fun refine(px: IntArray, conf: FloatArray, w: Int, h: Int): FloatArray {
        val r = maxOf(2, maxOf(w, h) / 128)
        val g = guided(Pixels.lumaF(px), conf, w, h, r, 1e-3f)
        for (i in g.indices) g[i] = Pixels.smoothstep(0.4f, 0.8f, g[i])
        return g
    }

    /** 有没有人：蒙版里足够确定的像素占比 */
    fun coverage(mask: FloatArray): Float {
        var n = 0
        for (m in mask) if (m > 0.5f) n++
        return n.toFloat() / mask.size.coerceAtLeast(1)
    }

    /** 透明背景：alpha 乘以蒙版，颜色不动（不会有黑边） */
    fun transparent(px: IntArray, mask: FloatArray): IntArray = IntArray(px.size) {
        val c = px[it]
        val a = Pixels.clamp255((c ushr 24) * mask[it])
        (a shl 24) or (c and 0xffffff)
    }

    /** 换成纯色背景 */
    fun overColor(px: IntArray, mask: FloatArray, color: Int): IntArray {
        val bg = IntArray(px.size) { color or Pixels.ALPHA }
        return Pixels.composite(px, bg, mask)
    }

    /** 把人像区域标成要填的洞（略微外扩），用来在虚化前先把人从背景里抹掉，避免人像颜色晕进背景 */
    fun holeOf(mask: FloatArray, w: Int, h: Int): FloatArray {
        val grown = Pixels.blurF(mask, w, h, maxOf(2, maxOf(w, h) / 100), 2)
        return FloatArray(mask.size) { if (grown[it] > 0.08f || mask[it] > 0.3f) 1f else 0f }
    }
}
