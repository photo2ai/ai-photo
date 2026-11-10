package com.photo2ai.aiphoto.core

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * 精细人像抠图：MODNet（人像 matting，Apache 2.0）出头发丝级别的透明度，再用导向滤波按原图边缘细化。
 * ML Kit 自拍分割的输入只有 256×256，发丝、飘起来的头发会被一刀切掉；MODNet 在 512 上算（它就是按 512 训练的，
 * 放大到 768、1024 反而会把衣服挖出洞）。
 */
object Matting {
    /** 透明度在长边这么大的图上细化，确认时按原图尺寸放大 */
    const val SIDE = 2048

    /** MODNet 的输入尺寸：短边缩到 512（本来就在 512 附近的不缩），两边都取 32 的倍数（和官方推理脚本一样） */
    fun inputSize(w: Int, h: Int, ref: Int = 512): Pair<Int, Int> {
        var rw = w
        var rh = h
        if (max(w, h) < ref || minOf(w, h) > ref) {
            if (w >= h) {
                rh = ref
                rw = (w.toFloat() / h * ref).toInt()
            } else {
                rw = ref
                rh = (h.toFloat() / w * ref).toInt()
            }
        }
        return (rw - rw % 32) to (rh - rh % 32)
    }

    /** 人像透明度（0..1），尺寸是 [src] 缩到长边 [SIDE] 后的大小 */
    fun person(model: File, src: Bitmap): Segment.Mask {
        val mid = ImageIO.fit(src, SIDE)
        val w = mid.width
        val h = mid.height
        val (rw, rh) = inputSize(w, h)
        val small = ImageIO.pixels(ImageIO.scaled(mid, rw, rh))
        val n = rw * rh
        val input = FloatArray(3 * n)
        for (i in 0 until n) {
            val c = small[i]
            input[i] = (c shr 16 and 0xff) / 127.5f - 1f
            input[n + i] = (c shr 8 and 0xff) / 127.5f - 1f
            input[2 * n + i] = (c and 0xff) / 127.5f - 1f
        }
        val s = LocalAi.session(model)
        val raw = OnnxTensor.createTensor(LocalAi.env, FloatBuffer.wrap(input), longArrayOf(1, 3, rh.toLong(), rw.toLong())).use { t ->
            s.run(mapOf(s.inputNames.first() to t)).use { r ->
                val fb = (r.get(0) as OnnxTensor).floatBuffer
                FloatArray(fb.remaining()).also { fb.get(it) }
            }
        }
        return Segment.Mask(refine(ImageIO.pixels(mid), Pixels.resizeF(raw, rw, rh, w, h), w, h), w, h)
    }

    /**
     * 按图像边缘细化透明度：小半径、小 eps 的导向滤波（以亮度为引导），让衣服、肩膀的轮廓贴到真实边缘上，
     * 发丝这种细节保持 MODNet 原样（电脑上比过：比不细化的轮廓清楚一点，发丝不丢）。再把过渡带收到 0.2～0.8：
     * 头发周围透明度很低的一层「雾」（原图里飘动的头发是糊的）换白底时发灰，算作背景；发丝本身透明度高，留着。
     * ML Kit 那边用的是 0.4～0.8，会把两侧的头发切掉一大块
     */
    fun refine(px: IntArray, alpha: FloatArray, w: Int, h: Int): FloatArray {
        val g = Cutout.guided(Pixels.lumaF(px), alpha, w, h, max(2, max(w, h) / 512), 1e-5f)
        for (i in g.indices) g[i] = Pixels.smoothstep(0.2f, 0.8f, g[i])
        return g
    }

    /**
     * 前景颜色估计（Forte & Pitié 2021 的 blur fusion）：边缘半透明的像素是「人」和「原背景」混在一起的颜色，直接换底
     * 会在头发周围留一圈原背景色的光晕。用模糊估出附近的纯前景色、纯背景色，把混进去的背景色减掉。
     * 先大半径（脸宽量级）再小半径各算一遍；返回前景颜色（alpha 照搬 [px]）。
     */
    fun foreground(px: IntArray, w: Int, h: Int, alpha: FloatArray): IntArray {
        val n = w * h
        val scale = max(w, h) / 1024f
        val rBig = max(1, (90 * scale).roundToInt())
        val rSmall = max(1, (6 * scale).roundToInt())
        val inv = FloatArray(n) { 1f - alpha[it] }
        val out = IntArray(n) { px[it] and Pixels.ALPHA }
        for (sh in intArrayOf(16, 8, 0)) {
            val img = FloatArray(n) { (px[it] shr sh and 0xff) / 255f }
            var f = img
            var b = img
            for (r in intArrayOf(rBig, rSmall)) {
                val ba = Pixels.blurF(alpha, w, h, r, 1)
                val bfa = Pixels.blurF(FloatArray(n) { f[it] * alpha[it] }, w, h, r, 1)
                val bba = Pixels.blurF(FloatArray(n) { b[it] * inv[it] }, w, h, r, 1)
                val nf = FloatArray(n)
                val nb = FloatArray(n)
                for (i in 0 until n) {
                    val a = alpha[i]
                    val fBlur = bfa[i] / (ba[i] + 1e-5f)
                    val bBlur = bba[i] / ((1f - ba[i]) + 1e-5f)
                    nf[i] = (fBlur + a * (img[i] - a * fBlur - (1f - a) * bBlur)).coerceIn(0f, 1f)
                    nb[i] = bBlur
                }
                f = nf
                b = nb
            }
            for (i in 0 until n) out[i] = out[i] or (Pixels.clamp255(f[i] * 255f) shl sh)
        }
        return out
    }

    /**
     * 原图尺寸的前景颜色：在长边 [SIDE] 的图上估，把「前景 − 原图」的差按双线性放大加到原图上（只改半透明的边缘，
     * 不用整张原图尺寸的浮点数组，1600 万像素的图也不会占太多内存）
     */
    fun foregroundFull(src: Bitmap, mask: Segment.Mask): IntArray {
        val px = ImageIO.pixels(src)
        val w = src.width
        val h = src.height
        val mid = if (max(w, h) <= SIDE) src else ImageIO.fit(src, SIDE)
        val mw = mid.width
        val mh = mid.height
        val mpx = if (mid === src) px else ImageIO.pixels(mid)
        val ma = mask.resized(mw, mh)
        val fg = foreground(mpx, mw, mh, ma)
        if (mid === src) return fg
        val full = mask.resized(w, h)
        val d = Array(3) { ch -> val sh = 16 - 8 * ch; FloatArray(mw * mh) { ((fg[it] shr sh and 0xff) - (mpx[it] shr sh and 0xff)).toFloat() } }
        val out = px.copyOf()
        val kx = mw.toFloat() / w
        val ky = mh.toFloat() / h
        for (y in 0 until h) {
            val fy = ((y + 0.5f) * ky - 0.5f).coerceIn(0f, (mh - 1).toFloat())
            val y0 = fy.toInt()
            val y1 = minOf(y0 + 1, mh - 1)
            val ty = fy - y0
            for (x in 0 until w) {
                val i = y * w + x
                val a = full[i]
                if (a <= 0.003f || a >= 0.997f) continue
                val fx = ((x + 0.5f) * kx - 0.5f).coerceIn(0f, (mw - 1).toFloat())
                val x0 = fx.toInt()
                val x1 = minOf(x0 + 1, mw - 1)
                val tx = fx - x0
                val c = px[i]
                var r = c and Pixels.ALPHA
                for (ch in 0..2) {
                    val dd = d[ch]
                    val top = dd[y0 * mw + x0] * (1 - tx) + dd[y0 * mw + x1] * tx
                    val bot = dd[y1 * mw + x0] * (1 - tx) + dd[y1 * mw + x1] * tx
                    val sh = 16 - 8 * ch
                    r = r or (Pixels.clamp255((c shr sh and 0xff) + top * (1 - ty) + bot * ty) shl sh)
                }
                out[i] = r
            }
        }
        return out
    }
}
