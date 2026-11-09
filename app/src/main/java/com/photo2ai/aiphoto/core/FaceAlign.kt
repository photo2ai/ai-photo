package com.photo2ai.aiphoto.core

import kotlin.math.max
import kotlin.math.min

/**
 * 人脸对齐（纯计算）：按 5 个关键点（左眼、右眼、鼻尖、左嘴角、右嘴角，图像里的左右）求相似变换，
 * 把脸摆正、缩放到模型要的标准位置。仿射矩阵记作 [a, b, tx, c, d, ty]：x' = a·x + b·y + tx，y' = c·x + d·y + ty。
 */
object FaceAlign {
    /** ArcFace 112×112 的标准关键点位置 */
    val ARCFACE_112 = floatArrayOf(
        38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f, 71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f,
    )

    /** ArcFace 128×128 的标准位置：ArcFace 模板整体右移 8 像素（和 insightface 的 norm_crop、FaceFusion 的 arcface_128 一致；HyperSwap 按它放大到 256） */
    val ARCFACE_128 = FloatArray(10) { if (it % 2 == 0) ARCFACE_112[it] + 8f else ARCFACE_112[it] }

    /** GFPGAN 512×512 的标准位置（FFHQ 数据集的对齐方式，和 FaceFusion 一致） */
    val FFHQ_512 = floatArrayOf(
        0.37691676f, 0.46864664f, 0.62285697f, 0.46912813f, 0.50123859f, 0.61331904f, 0.39308822f, 0.72541100f, 0.61150205f, 0.72490465f,
    ).let { t -> FloatArray(t.size) { t[it] * 512f } }

    fun scaled(t: FloatArray, k: Float) = FloatArray(t.size) { t[it] * k }

    /** 最小二乘相似变换（旋转 + 等比缩放 + 平移），把 [src] 的点映射到 [dst] */
    fun similarity(src: FloatArray, dst: FloatArray): FloatArray {
        val n = src.size / 2
        var sx = 0f; var sy = 0f; var dx = 0f; var dy = 0f
        for (i in 0 until n) {
            sx += src[2 * i]; sy += src[2 * i + 1]; dx += dst[2 * i]; dy += dst[2 * i + 1]
        }
        sx /= n; sy /= n; dx /= n; dy /= n
        var num1 = 0f; var num2 = 0f; var den = 0f
        for (i in 0 until n) {
            val x = src[2 * i] - sx
            val y = src[2 * i + 1] - sy
            val u = dst[2 * i] - dx
            val v = dst[2 * i + 1] - dy
            num1 += x * u + y * v
            num2 += x * v - y * u
            den += x * x + y * y
        }
        val a = num1 / den
        val b = num2 / den
        return floatArrayOf(a, -b, dx - (a * sx - b * sy), b, a, dy - (b * sx + a * sy))
    }

    fun invert(m: FloatArray): FloatArray {
        val det = m[0] * m[4] - m[1] * m[3]
        val a = m[4] / det
        val b = -m[1] / det
        val c = -m[3] / det
        val d = m[0] / det
        return floatArrayOf(a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5]))
    }

    fun apply(m: FloatArray, x: Float, y: Float) = floatArrayOf(m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])

    /** 按 [m]（原图 → 输出）把原图变换成 ow×oh 的输出图：输出的每个像素反算回原图双线性取色，出界取边缘 */
    fun warp(px: IntArray, w: Int, h: Int, m: FloatArray, ow: Int, oh: Int): IntArray {
        val inv = invert(m)
        val out = IntArray(ow * oh)
        for (y in 0 until oh) for (x in 0 until ow) {
            val fx = inv[0] * x + inv[1] * y + inv[2]
            val fy = inv[3] * x + inv[4] * y + inv[5]
            out[y * ow + x] = sample(px, w, h, fx, fy)
        }
        return out
    }

    fun sample(px: IntArray, w: Int, h: Int, fx: Float, fy: Float): Int {
        val x = fx.coerceIn(0f, (w - 1).toFloat())
        val y = fy.coerceIn(0f, (h - 1).toFloat())
        val x0 = x.toInt()
        val y0 = y.toInt()
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)
        val top = Pixels.lerp(px[y0 * w + x0], px[y0 * w + x1], x - x0)
        val bot = Pixels.lerp(px[y1 * w + x0], px[y1 * w + x1], x - x0)
        return Pixels.lerp(top, bot, y - y0)
    }

    /** 对齐后的脸要贴回原图的范围（原图坐标，含边）：标准框四个角变换回原图后的外接矩形 */
    fun backBounds(m: FloatArray, size: Int, w: Int, h: Int): IntArray {
        val inv = invert(m)
        var x0 = Float.MAX_VALUE; var y0 = Float.MAX_VALUE; var x1 = -Float.MAX_VALUE; var y1 = -Float.MAX_VALUE
        for ((cx, cy) in listOf(0f to 0f, size.toFloat() to 0f, 0f to size.toFloat(), size.toFloat() to size.toFloat())) {
            val p = apply(inv, cx, cy)
            x0 = min(x0, p[0]); y0 = min(y0, p[1]); x1 = max(x1, p[0]); y1 = max(y1, p[1])
        }
        return intArrayOf(max(0, x0.toInt()), max(0, y0.toInt()), min(w, x1.toInt() + 1), min(h, y1.toInt() + 1))
    }
}
