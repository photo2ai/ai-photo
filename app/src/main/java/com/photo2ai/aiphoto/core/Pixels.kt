package com.photo2ai.aiphoto.core

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 纯 Kotlin 的像素运算。像素是非预乘的 ARGB Int（Bitmap.getPixels 的格式），不依赖 android.graphics，能在 JVM 上单测。
 */
object Pixels {
    const val ALPHA = -0x1000000

    fun clamp255(v: Int) = if (v < 0) 0 else if (v > 255) 255 else v
    fun clamp255(v: Float) = if (v <= 0f) 0 else if (v >= 255f) 255 else (v + 0.5f).toInt()
    fun rgb(alphaOf: Int, r: Int, g: Int, b: Int) = (alphaOf and ALPHA) or (r shl 16) or (g shl 8) or b

    /** 0..255 的亮度 */
    fun luma(c: Int): Int = ((c shr 16 and 0xff) * 77 + (c shr 8 and 0xff) * 150 + (c and 0xff) * 29) shr 8

    fun smoothstep(e0: Float, e1: Float, x: Float): Float {
        val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f)
        return t * t * (3 - 2 * t)
    }

    /** 盒式模糊（横竖各一遍滑动窗口，passes 遍近似高斯）。只模糊 RGB，alpha 照搬原图 */
    fun boxBlur(src: IntArray, w: Int, h: Int, radius: Int, passes: Int = 3): IntArray {
        if (radius <= 0 || w == 0 || h == 0) return src.copyOf()
        val a = src.copyOf()
        val tmp = IntArray(src.size)
        repeat(passes) {
            blurLine(a, tmp, w, h, radius, horizontal = true)
            blurLine(tmp, a, w, h, radius, horizontal = false)
        }
        return a
    }

    private fun blurLine(inp: IntArray, out: IntArray, w: Int, h: Int, r: Int, horizontal: Boolean) {
        val len = if (horizontal) w else h
        val lines = if (horizontal) h else w
        val step = if (horizontal) 1 else w
        val div = 2 * r + 1
        val half = div / 2
        for (line in 0 until lines) {
            val start = if (horizontal) line * w else line
            var sr = 0
            var sg = 0
            var sb = 0
            for (k in -r..r) {
                val c = inp[start + min(max(k, 0), len - 1) * step]
                sr += c shr 16 and 0xff; sg += c shr 8 and 0xff; sb += c and 0xff
            }
            for (i in 0 until len) {
                val idx = start + i * step
                out[idx] = (inp[idx] and ALPHA) or (((sr + half) / div) shl 16) or (((sg + half) / div) shl 8) or ((sb + half) / div)
                val add = inp[start + min(i + r + 1, len - 1) * step]
                val rem = inp[start + max(i - r, 0) * step]
                sr += (add shr 16 and 0xff) - (rem shr 16 and 0xff)
                sg += (add shr 8 and 0xff) - (rem shr 8 and 0xff)
                sb += (add and 0xff) - (rem and 0xff)
            }
        }
    }

    /** 单通道浮点的盒式模糊（蒙版、亮度用） */
    fun blurF(src: FloatArray, w: Int, h: Int, radius: Int, passes: Int = 3): FloatArray {
        if (radius <= 0 || w == 0 || h == 0) return src.copyOf()
        val a = src.copyOf()
        val tmp = FloatArray(src.size)
        repeat(passes) {
            blurLineF(a, tmp, w, h, radius, true)
            blurLineF(tmp, a, w, h, radius, false)
        }
        return a
    }

    private fun blurLineF(inp: FloatArray, out: FloatArray, w: Int, h: Int, r: Int, horizontal: Boolean) {
        val len = if (horizontal) w else h
        val lines = if (horizontal) h else w
        val step = if (horizontal) 1 else w
        val inv = 1f / (2 * r + 1)
        for (line in 0 until lines) {
            val start = if (horizontal) line * w else line
            var s = 0f
            for (k in -r..r) s += inp[start + min(max(k, 0), len - 1) * step]
            for (i in 0 until len) {
                out[start + i * step] = s * inv
                s += inp[start + min(i + r + 1, len - 1) * step] - inp[start + max(i - r, 0) * step]
            }
        }
    }

    /** 非锐化掩模：原图 + amount ×（原图 − 模糊） */
    fun sharpen(px: IntArray, w: Int, h: Int, amount: Float, radius: Int = 1): IntArray {
        if (amount <= 0f) return px
        val blur = boxBlur(px, w, h, radius, 1)
        val out = IntArray(px.size)
        for (i in px.indices) {
            val c = px[i]
            val b = blur[i]
            val r = c shr 16 and 0xff
            val g = c shr 8 and 0xff
            val bl = c and 0xff
            out[i] = rgb(
                c,
                clamp255(r + amount * (r - (b shr 16 and 0xff))),
                clamp255(g + amount * (g - (b shr 8 and 0xff))),
                clamp255(bl + amount * (bl - (b and 0xff))),
            )
        }
        return out
    }

    /** 暗角：amount < 0 压暗四角，> 0 提亮四角 */
    fun vignette(px: IntArray, w: Int, h: Int, amount: Float): IntArray {
        if (amount == 0f) return px
        val out = IntArray(px.size)
        val cx = w / 2f
        val cy = h / 2f
        val maxD = sqrt(cx * cx + cy * cy)
        for (y in 0 until h) {
            val dy = (y - cy) / maxD
            for (x in 0 until w) {
                val dx = (x - cx) / maxD
                val f = smoothstep(0.35f, 1.0f, sqrt(dx * dx + dy * dy))
                val i = y * w + x
                val c = px[i]
                if (f == 0f) {
                    out[i] = c
                    continue
                }
                var r = (c shr 16 and 0xff).toFloat()
                var g = (c shr 8 and 0xff).toFloat()
                var b = (c and 0xff).toFloat()
                if (amount < 0) {
                    val k = 1f + amount * 0.8f * f
                    r *= k; g *= k; b *= k
                } else {
                    val k = amount * 0.6f * f
                    r += (255 - r) * k; g += (255 - g) * k; b += (255 - b) * k
                }
                out[i] = rgb(c, clamp255(r), clamp255(g), clamp255(b))
            }
        }
        return out
    }

    /** 从 a 到 b 的线性插值，t = 0 是 a、1 是 b（强度滑杆用） */
    fun mix(a: IntArray, b: IntArray, t: Float): IntArray {
        if (t >= 1f) return b.copyOf()
        if (t <= 0f) return a.copyOf()
        val out = IntArray(a.size)
        for (i in a.indices) out[i] = lerp(a[i], b[i], t)
        return out
    }

    /** 按蒙版混合：m = 1 取 fg，0 取 bg（alpha 也一起插值） */
    fun composite(fg: IntArray, bg: IntArray, mask: FloatArray): IntArray {
        val out = IntArray(fg.size)
        for (i in fg.indices) {
            val m = mask[i]
            out[i] = if (m >= 1f) fg[i] else if (m <= 0f) bg[i] else lerp(bg[i], fg[i], m)
        }
        return out
    }

    fun lerp(a: Int, b: Int, t: Float): Int {
        val aa = a ushr 24
        val ar = a shr 16 and 0xff
        val ag = a shr 8 and 0xff
        val ab = a and 0xff
        return (clamp255(aa + (((b ushr 24) - aa) * t)) shl 24) or
            (clamp255(ar + (((b shr 16 and 0xff) - ar) * t)) shl 16) or
            (clamp255(ag + (((b shr 8 and 0xff) - ag) * t)) shl 8) or
            clamp255(ab + (((b and 0xff) - ab) * t))
    }

    fun gray(px: IntArray): IntArray = IntArray(px.size) { val l = luma(px[it]); rgb(px[it], l, l, l) }

    /** 0..1 的亮度图 */
    fun lumaF(px: IntArray): FloatArray = FloatArray(px.size) { luma(px[it]) / 255f }

    /** 双线性缩放单通道浮点图 */
    fun resizeF(src: FloatArray, sw: Int, sh: Int, dw: Int, dh: Int): FloatArray {
        if (sw == dw && sh == dh) return src.copyOf()
        val out = FloatArray(dw * dh)
        val fx = sw.toFloat() / dw
        val fy = sh.toFloat() / dh
        for (y in 0 until dh) {
            val sy = ((y + 0.5f) * fy - 0.5f).coerceIn(0f, (sh - 1).toFloat())
            val y0 = sy.toInt()
            val y1 = min(y0 + 1, sh - 1)
            val ty = sy - y0
            for (x in 0 until dw) {
                val sx = ((x + 0.5f) * fx - 0.5f).coerceIn(0f, (sw - 1).toFloat())
                val x0 = sx.toInt()
                val x1 = min(x0 + 1, sw - 1)
                val tx = sx - x0
                val top = src[y0 * sw + x0] * (1 - tx) + src[y0 * sw + x1] * tx
                val bot = src[y1 * sw + x0] * (1 - tx) + src[y1 * sw + x1] * tx
                out[y * dw + x] = top * (1 - ty) + bot * ty
            }
        }
        return out
    }

    /** 有没有不完全不透明的像素（决定存 PNG 还是 JPEG） */
    fun hasTransparency(px: IntArray): Boolean {
        for (c in px) if ((c ushr 24) < 255) return true
        return false
    }
}
