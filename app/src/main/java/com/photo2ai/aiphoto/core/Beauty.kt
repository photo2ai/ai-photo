package com.photo2ai.aiphoto.core

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * 人像美颜（纯计算，可单测）：磨皮、美白只作用在脸部皮肤上（避开眼、眉、嘴），瘦脸、大眼是局部液化变形。
 * 半径、力度都按脸宽算，所以预览图和原图上的效果一致。
 */
object Beauty {
    /** 各项强度 0..1 */
    data class Params(val smooth: Float = 0f, val whiten: Float = 0f, val slim: Float = 0f, val eyes: Float = 0f) {
        val isIdentity get() = smooth <= 0f && whiten <= 0f && slim <= 0f && eyes <= 0f
    }

    /** 一张脸：网格点（x,y 交替，图像坐标）、外轮廓编号、两只眼睛的轮廓编号、不磨皮美白的五官轮廓（眼、眉、嘴） */
    class Geom(val pts: FloatArray, val oval: IntArray, val eyes: List<IntArray>, val features: List<IntArray>) {
        fun scaled(k: Float) = Geom(FloatArray(pts.size) { pts[it] * k }, oval, eyes, features)
    }

    /** 一张脸预先算好的东西，只和照片、脸有关，拖滑杆时不用重算。坐标都在脸周围裁出的这一块里 */
    class Prepared internal constructor(
        val x0: Int, val y0: Int, val w: Int, val h: Int,
        internal val src: IntArray,
        internal val smoothed: IntArray,
        /** 磨皮蒙版：脸内的皮肤 */
        internal val skin: FloatArray,
        /** 美白蒙版：脸内的皮肤，往外（脖子、耳朵）缓慢淡出，免得脸和脖子之间出现分界 */
        internal val light: FloatArray,
        /** 每只眼睛（中心 x、y，半径） */
        internal val eyes: List<FloatArray>,
        /** 瘦脸的锚点（x、y，强度 100% 时往里推的位移 dx、dy）和影响半径 */
        internal val jaw: FloatArray,
        internal val jawR: Float,
    )

    /** 脸宽小于这个就不处理（太小看不出效果，网格也不准） */
    private const val MIN_FACE = 24f

    fun prepare(px: IntArray, w: Int, h: Int, g: Geom): Prepared? {
        val poly = polygon(g.pts, g.oval)
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE; var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (i in 0 until poly.size / 2) {
            minX = min(minX, poly[2 * i]); maxX = max(maxX, poly[2 * i])
            minY = min(minY, poly[2 * i + 1]); maxY = max(maxY, poly[2 * i + 1])
        }
        val fw = maxX - minX
        if (fw < MIN_FACE || g.eyes.size < 2) return null
        // 往外留半个脸宽：瘦脸会拉动下颌外侧的背景，美白要淡出到脖子
        val m = fw * 0.5f
        val x0 = max(0, (minX - m).toInt())
        val y0 = max(0, (minY - m * 0.6f).toInt())
        val x1 = min(w, (maxX + m).toInt() + 1)
        val y1 = min(h, (maxY + m).toInt() + 1)
        val cw = x1 - x0
        val ch = y1 - y0
        if (cw < 8 || ch < 8) return null
        val src = IntArray(cw * ch)
        for (y in 0 until ch) System.arraycopy(px, (y0 + y) * w + x0, src, y * cw, cw)
        val pts = FloatArray(g.pts.size) { if (it % 2 == 0) g.pts[it] - x0 else g.pts[it] - y0 }

        val oval = FaceSwap.fill(polygon(pts, g.oval), cw, ch)
        // 五官往外扩一圈（睫毛、眉毛边缘、唇线不能被磨糊），再羽化
        var feat = FloatArray(cw * ch)
        for (f in g.features) {
            val m1 = FaceSwap.fill(polygon(pts, f), cw, ch)
            for (i in feat.indices) if (m1[i] > 0f) feat[i] = 1f
        }
        feat = Pixels.blurF(feat, cw, ch, max(1, (fw / 50).roundToInt()), 1)
        for (i in feat.indices) feat[i] = if (feat[i] > 0.02f) 1f else 0f
        feat = Pixels.blurF(feat, cw, ch, max(1, (fw / 80).roundToInt()), 2)

        val skinLike = skinLikelihood(src, oval, feat, cw, ch, fw)

        val skin = Pixels.blurF(oval, cw, ch, max(1, (fw / 30).roundToInt()), 2)
        for (i in skin.indices) skin[i] = Pixels.smoothstep(0.2f, 0.8f, skin[i]) * skinLike[i] * (1f - feat[i])
        val light = Pixels.blurF(oval, cw, ch, max(1, (fw / 6).roundToInt()), 2)
        for (i in light.indices) light[i] = min(1f, light[i] * 1.6f) * skinLike[i] * (1f - 0.8f * feat[i])

        val smoothed = smoothSkin(src, cw, ch, fw)

        val eyes = g.eyes.take(2).map { idx -> eyeCircle(pts, idx) }
        val jawR = fw * 0.22f
        return Prepared(x0, y0, cw, ch, src, smoothed, skin, light, eyes, jawAnchors(pts, g.oval, eyes, fw), jawR)
    }

    /** 按 [p] 处理所有的脸，返回整张图（[px] 不改） */
    fun apply(px: IntArray, w: Int, faces: List<Prepared>, p: Params): IntArray {
        val out = px.copyOf()
        for (f in faces) {
            val r = render(f, p)
            // 只写有变化的像素：几张脸裁出的块会重叠，不能拿一张脸的原样像素盖掉另一张脸的结果
            for (y in 0 until f.h) {
                val row = (f.y0 + y) * w + f.x0
                for (x in 0 until f.w) {
                    val i = y * f.w + x
                    if (r[i] != f.src[i]) out[row + x] = r[i]
                }
            }
        }
        return out
    }

    /** 一张脸那一块的结果 */
    fun render(f: Prepared, p: Params): IntArray {
        val n = f.w * f.h
        val out = f.src.copyOf()
        if (p.smooth > 0f) {
            // 满强度也留一成原图的纹理，免得像塑料
            val s = p.smooth * 0.9f
            for (i in 0 until n) {
                val t = f.skin[i] * s
                if (t > 0.004f) out[i] = Pixels.lerp(f.src[i], f.smoothed[i], t)
            }
        }
        if (p.whiten > 0f) {
            // 对数曲线提亮：暗部提得多、亮部几乎不动，不会过曝
            val beta = 1f + 3f * p.whiten
            val lut = IntArray(256) { Pixels.clamp255(255f * ln(1f + (beta - 1f) * it / 255f) / ln(beta)) }
            for (i in 0 until n) {
                val t = f.light[i]
                if (t <= 0.004f) continue
                val c = out[i]
                val r = c shr 16 and 0xff; val gg = c shr 8 and 0xff; val b = c and 0xff
                out[i] = Pixels.rgb(c, mixI(r, lut[r], t), mixI(gg, lut[gg], t), mixI(b, lut[b], t))
            }
        }
        if (p.slim <= 0f && p.eyes <= 0f) return out
        return warp(out, f, p)
    }

    private fun mixI(a: Int, b: Int, t: Float) = (a + (b - a) * t + 0.5f).toInt()

    /**
     * 液化：先在每个像素上累加「从哪里取色」的偏移，再统一双线性取一次色。
     * 瘦脸：下颌两侧的锚点往脸中线推，几个锚点的影响按权重平均（不是相加，挨得近的锚点不会把力度叠成几倍）。
     * 大眼：眼睛中心附近从更靠近中心的地方取色，越靠中心放大越多，到半径处平滑回到原样。
     */
    private fun warp(img: IntArray, f: Prepared, p: Params): IntArray {
        val w = f.w
        val h = f.h
        val ox = FloatArray(w * h)
        val oy = FloatArray(w * h)
        if (p.slim > 0f) {
            val wsum = FloatArray(w * h)
            val r = f.jawR
            val r2 = r * r
            var k = 0
            while (k + 3 < f.jaw.size) {
                val cx = f.jaw[k]; val cy = f.jaw[k + 1]
                val dx = f.jaw[k + 2] * p.slim; val dy = f.jaw[k + 3] * p.slim
                k += 4
                for (y in max(0, (cy - r).toInt())..min(h - 1, (cy + r).toInt())) {
                    val py = y + 0.5f - cy
                    for (x in max(0, (cx - r).toInt())..min(w - 1, (cx + r).toInt())) {
                        val px = x + 0.5f - cx
                        val d2 = px * px + py * py
                        if (d2 >= r2) continue
                        val q = 1f - d2 / r2
                        val wt = q * q
                        val i = y * w + x
                        // 内容往里移 = 从外侧取色
                        ox[i] -= wt * dx
                        oy[i] -= wt * dy
                        wsum[i] += wt
                    }
                }
            }
            for (i in wsum.indices) if (wsum[i] > 1f) { ox[i] /= wsum[i]; oy[i] /= wsum[i] }
        }
        if (p.eyes > 0f) {
            val k = 0.22f * p.eyes
            for (e in f.eyes) {
                val cx = e[0]; val cy = e[1]; val r = e[2]
                val r2 = r * r
                for (y in max(0, (cy - r).toInt())..min(h - 1, (cy + r).toInt())) {
                    val py = y + 0.5f - cy
                    for (x in max(0, (cx - r).toInt())..min(w - 1, (cx + r).toInt())) {
                        val px = x + 0.5f - cx
                        val d2 = px * px + py * py
                        if (d2 >= r2) continue
                        val q = 1f - d2 / r2
                        val s = k * q * q
                        val i = y * w + x
                        ox[i] -= px * s
                        oy[i] -= py * s
                    }
                }
            }
        }
        val out = img.copyOf()
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            if (ox[i] == 0f && oy[i] == 0f) continue
            out[i] = FaceSwap.sample(img, w, h, x + ox[i], y + oy[i])
        }
        return out
    }

    /**
     * 磨皮底图：每个通道做自引导的导向滤波。斑点、痘印、毛孔这种小起伏被抹平，五官、轮廓这种大反差的边缘留住。
     * 半径按脸宽：预览图和原图上抹掉的是同样大小的东西。引导图就是自己，比通用的导向滤波少一半中间数组
     * （原图上脸很大时，裁出的这一块接近整张图）。
     */
    private fun smoothSkin(px: IntArray, w: Int, h: Int, fw: Float): IntArray {
        val r = max(2, (fw / 45).roundToInt())
        val eps = 0.006f
        val out = IntArray(px.size) { px[it] and Pixels.ALPHA }
        for (sh in intArrayOf(16, 8, 0)) {
            val ch = FloatArray(px.size) { (px[it] shr sh and 0xff) / 255f }
            val mean = Pixels.blurF(ch, w, h, r, 1)
            val sq = Pixels.blurF(FloatArray(px.size) { ch[it] * ch[it] }, w, h, r, 1)
            // 就地换成 a（放 sq 里）和 b（放 mean 里）
            for (i in mean.indices) {
                val m = mean[i]
                val v = sq[i] - m * m
                val a = v / (v + eps)
                sq[i] = a
                mean[i] = m - a * m
            }
            val ma = Pixels.blurF(sq, w, h, r, 1)
            val mb = Pixels.blurF(mean, w, h, r, 1)
            for (i in out.indices) out[i] = out[i] or (Pixels.clamp255((ma[i] * ch[i] + mb[i]) * 255f) shl sh)
        }
        return out
    }

    /**
     * 每个像素像不像这张脸的肤色（0..1）：在脸内（去掉五官）统计色度（Cb、Cr）的均值和离散程度，离均值越远越不像。
     * 头发、背景、眼镜框、衣领被排除在磨皮美白之外；黑白、泛黄的老照片色度几乎一样，整块都算皮肤，也没问题。
     */
    private fun skinLikelihood(px: IntArray, oval: FloatArray, feat: FloatArray, w: Int, h: Int, fw: Float): FloatArray {
        fun cb(c: Int) = -0.1687f * (c shr 16 and 0xff) - 0.3313f * (c shr 8 and 0xff) + 0.5f * (c and 0xff)
        fun cr(c: Int) = 0.5f * (c shr 16 and 0xff) - 0.4187f * (c shr 8 and 0xff) - 0.0813f * (c and 0xff)
        var sb = 0.0; var sr = 0.0; var cnt = 0
        for (i in px.indices) if (oval[i] > 0f && feat[i] < 0.5f) { sb += cb(px[i]); sr += cr(px[i]); cnt++ }
        val out = FloatArray(px.size)
        if (cnt == 0) return out
        val mb = (sb / cnt).toFloat(); val mr = (sr / cnt).toFloat()
        // 离散程度用均方根：脸内阴影、腮红都在里面
        var sd = 0.0
        for (i in px.indices) if (oval[i] > 0f && feat[i] < 0.5f) {
            val db = cb(px[i]) - mb; val dr = cr(px[i]) - mr
            sd += db * db + dr * dr
        }
        val sigma = (2f * sqrt((sd / cnt).toFloat())).coerceIn(8f, 25f)
        for (i in px.indices) {
            val db = cb(px[i]) - mb; val dr = cr(px[i]) - mr
            out[i] = 1f - Pixels.smoothstep(sigma, 2.5f * sigma, sqrt(db * db + dr * dr))
        }
        // 稍微模糊一下，避免噪点在蒙版里出现斑驳
        return Pixels.blurF(out, w, h, max(1, (fw / 100).roundToInt()), 1)
    }

    private fun eyeCircle(pts: FloatArray, idx: IntArray): FloatArray {
        var cx = 0f; var cy = 0f
        for (i in idx) { cx += pts[2 * i]; cy += pts[2 * i + 1] }
        cx /= idx.size; cy /= idx.size
        var far = 0f
        for (i in idx) {
            val dx = pts[2 * i] - cx; val dy = pts[2 * i + 1] - cy
            far = max(far, sqrt(dx * dx + dy * dy))
        }
        // far 约是眼长的一半；影响半径取眼长，眼皮、眼周一起平滑过渡
        return floatArrayOf(cx, cy, far * 2f)
    }

    /**
     * 瘦脸锚点：外轮廓上眼睛以下、下巴尖以上的点（从眼睛连线到下巴的 25%～90%），往脸中线推。
     * 推的力度在下颌中段最大，往颧骨和下巴尖逐渐减到 0，下巴尖本身不动。脸歪着也按眼睛连线的方向算。
     */
    internal fun jawAnchors(pts: FloatArray, oval: IntArray, eyes: List<FloatArray>, fw: Float): FloatArray {
        val ex = (eyes[0][0] + eyes[1][0]) / 2; val ey = (eyes[0][1] + eyes[1][1]) / 2
        var ax = eyes[1][0] - eyes[0][0]; var ay = eyes[1][1] - eyes[0][1]
        val al = sqrt(ax * ax + ay * ay).coerceAtLeast(1e-3f)
        ax /= al; ay /= al
        // 往下的方向：垂直于眼睛连线，指向轮廓的重心
        var ux = -ay; var uy = ax
        var gx = 0f; var gy = 0f
        for (i in oval) { gx += pts[2 * i]; gy += pts[2 * i + 1] }
        gx /= oval.size; gy /= oval.size
        if ((gx - ex) * ux + (gy - ey) * uy < 0) { ux = -ux; uy = -uy }
        var chin = 0f
        for (i in oval) chin = max(chin, (pts[2 * i] - ex) * ux + (pts[2 * i + 1] - ey) * uy)
        if (chin <= 0f) return FloatArray(0)
        val out = ArrayList<Float>()
        val mag = fw * 0.05f
        for (i in oval) {
            val dx = pts[2 * i] - ex; val dy = pts[2 * i + 1] - ey
            val t = (dx * ux + dy * uy) / chin
            if (t < 0.25f || t > 0.9f) continue
            val wt = 1f - kotlin.math.abs(t - 0.6f) / 0.35f
            if (wt <= 0f) continue
            val side = dx * ax + dy * ay
            val dir = if (side > 0) -1f else 1f
            out += pts[2 * i]; out += pts[2 * i + 1]
            out += dir * ax * mag * wt; out += dir * ay * mag * wt
        }
        return out.toFloatArray()
    }

    private fun polygon(pts: FloatArray, idx: IntArray) = FloatArray(idx.size * 2) { if (it % 2 == 0) pts[idx[it / 2] * 2] else pts[idx[it / 2] * 2 + 1] }
}
