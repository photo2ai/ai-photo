package com.photo2ai.aiphoto.core

import kotlin.math.max
import kotlin.math.min

/** 像素和模型张量之间的转换、分块（纯计算，可单测） */
object Tensors {
    /** ARGB → RGB 平面排列的 uint8（NCHW，N=1） */
    fun toChwU8(px: IntArray): ByteArray {
        val n = px.size
        val out = ByteArray(3 * n)
        for (i in 0 until n) {
            val c = px[i]
            out[i] = (c shr 16).toByte()
            out[n + i] = (c shr 8).toByte()
            out[2 * n + i] = c.toByte()
        }
        return out
    }

    /** RGB 平面 uint8 → ARGB，alpha 取自 [alphaFrom] */
    fun fromChwU8(b: ByteArray, alphaFrom: IntArray): IntArray {
        val n = alphaFrom.size
        return IntArray(n) {
            (alphaFrom[it] and Pixels.ALPHA) or ((b[it].toInt() and 0xff) shl 16) or
                ((b[n + it].toInt() and 0xff) shl 8) or (b[2 * n + it].toInt() and 0xff)
        }
    }

    /** MI-GAN 的蒙版：0 是要补的地方，255 保留。蒙版值 > [threshold] 算要补 */
    fun miganMask(mask: FloatArray, threshold: Float = 0.04f): ByteArray =
        ByteArray(mask.size) { if (mask[it] > threshold) 0 else 255.toByte() }

    /** 取一块 ARGB → RGB 平面 float 0..1 */
    fun tileToChwF32(px: IntArray, w: Int, x0: Int, y0: Int, tw: Int, th: Int): FloatArray {
        val n = tw * th
        val out = FloatArray(3 * n)
        for (y in 0 until th) {
            val row = (y0 + y) * w + x0
            for (x in 0 until tw) {
                val c = px[row + x]
                val i = y * tw + x
                out[i] = (c shr 16 and 0xff) / 255f
                out[n + i] = (c shr 8 and 0xff) / 255f
                out[2 * n + i] = (c and 0xff) / 255f
            }
        }
        return out
    }

    /**
     * 一块。输入里 [x],[y],[w],[h] 是送进模型的范围（含重叠边），[cx],[cy],[cw],[ch] 是这块实际负责写回的范围（不含重叠）。
     */
    data class Tile(val x: Int, val y: Int, val w: Int, val h: Int, val cx: Int, val cy: Int, val cw: Int, val ch: Int)

    /** 把 w×h 切成不超过 [size] 的块，每块四周多带 [overlap] 像素，拼接时只取中间部分，接缝不明显 */
    fun tiles(w: Int, h: Int, size: Int, overlap: Int): List<Tile> {
        val out = ArrayList<Tile>()
        var cy = 0
        while (cy < h) {
            val ch = min(size, h - cy)
            var cx = 0
            while (cx < w) {
                val cw = min(size, w - cx)
                val x = max(0, cx - overlap)
                val y = max(0, cy - overlap)
                out += Tile(x, y, min(w, cx + cw + overlap) - x, min(h, cy + ch + overlap) - y, cx, cy, cw, ch)
                cx += cw
            }
            cy += ch
        }
        return out
    }

    /**
     * 把一块放大 [scale] 倍的结果（RGB 平面 float）写回输出图，只写这块负责的中间部分。
     * [alphaFrom] 是原图像素，透明度按最近邻放大。
     */
    fun putScaledTile(out: IntArray, outW: Int, t: Tile, data: FloatArray, scale: Int, alphaFrom: IntArray, srcW: Int) {
        val tw = t.w * scale
        val n = tw * t.h * scale
        for (y in 0 until t.ch * scale) {
            val ty = (t.cy - t.y) * scale + y
            val oy = t.cy * scale + y
            for (x in 0 until t.cw * scale) {
                val tx = (t.cx - t.x) * scale + x
                val i = ty * tw + tx
                val ox = t.cx * scale + x
                val a = alphaFrom[(oy / scale) * srcW + ox / scale] and Pixels.ALPHA
                out[oy * outW + ox] = a or (Pixels.clamp255(data[i] * 255f) shl 16) or
                    (Pixels.clamp255(data[n + i] * 255f) shl 8) or Pixels.clamp255(data[2 * n + i] * 255f)
            }
        }
    }
}
