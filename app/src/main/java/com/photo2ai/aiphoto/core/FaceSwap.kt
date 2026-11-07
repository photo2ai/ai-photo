package com.photo2ai.aiphoto.core

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 本地换脸（纯计算）：两张脸用同一套人脸网格关键点，按三角形把参考脸逐块变形贴到目标脸的位置，
 * 再按目标脸的肤色调色，最后沿脸部轮廓羽化融合。只换五官和脸部皮肤，发型、轮廓外的东西都是目标照片的。
 */
object FaceSwap {
    /**
     * [src]（参考脸）贴到 [dst]（目标照片的一块）上，返回新的 dst 像素。
     * 点坐标是 x0,y0,x1,y1…，两边编号一致；[tris] 每三个编号一个三角形；[oval] 是脸部外轮廓按顺序的编号；
     * [holes] 是不换的区域（眼睛、张开的嘴）的轮廓编号：两张脸表情不同时，把眯着的眼睛拉成睁开的会变成两团黑，
     * 所以这些地方保留目标照片自己的。
     */
    fun swap(
        src: IntArray, sw: Int, sh: Int, srcPts: FloatArray,
        dst: IntArray, dw: Int, dh: Int, dstPts: FloatArray,
        tris: IntArray, oval: IntArray, holes: List<IntArray> = emptyList(),
    ): IntArray {
        val (warped, cover) = warp(src, sw, sh, srcPts, dw, dh, dstPts, tris)
        val ovalMask = faceMask(polygon(dstPts, oval), dw, dh)
        val mask = ovalMask.copyOf()
        // 眼睛、嘴：填充后稍微外扩羽化，从蒙版里扣掉
        val faceWidth = maxOf(8f, bounds(polygon(dstPts, oval))[2])
        for (h in holes) {
            val hole = fill(polygon(dstPts, h), dw, dh)
            val soft = Pixels.blurF(hole, dw, dh, maxOf(2, (faceWidth / 40f).toInt()), 2)
            for (i in mask.indices) mask[i] *= 1f - (soft[i] * 2.5f).coerceAtMost(1f)
        }
        for (i in mask.indices) mask[i] *= cover[i]
        // 肤色统计只看皮肤：蒙版里足够确定的部分（已经扣掉眼睛和嘴）
        val matched = matchColor(warped, dst, mask)
        // 光照迁移：明暗的大轮廓（哪边亮、哪边有阴影）换成目标照片的，五官和皮肤的细节保留参考脸的
        val colored = relight(matched, dst, ovalMask, cover, dw, dh, maxOf(3, (faceWidth / 8f).toInt()))
        // 色差修正只取外轮廓那一圈（不能取眼睛、嘴的边：那里是唇色、牙齿和皮肤的差，会把怪颜色扩散到整张脸）
        val ring = FloatArray(mask.size) { if (cover[it] > 0f) ovalMask[it] else 0f }
        val corrected = seamCorrect(colored, dst, ring, mask, dw, dh)
        return Pixels.composite(corrected, dst, mask)
    }

    /**
     * 光照迁移：两边亮度各做「只在脸上算」的大半径模糊（归一化卷积，避免把头发、背景混进来），
     * 差值加到贴上去的脸上。效果是低频的明暗跟目标走，高频的细节不变。
     */
    fun relight(face: IntArray, target: IntArray, region: FloatArray, cover: FloatArray, w: Int, h: Int, r: Int): IntArray {
        val m = FloatArray(face.size) { if (region[it] > 0.01f && cover[it] > 0f) 1f else 0f }
        val ys = FloatArray(face.size) { Pixels.luma(face[it]) * m[it] }
        val yt = FloatArray(face.size) { Pixels.luma(target[it]) * m[it] }
        val bm = Pixels.blurF(m, w, h, r, 3)
        val bs = Pixels.blurF(ys, w, h, r, 3)
        val bt = Pixels.blurF(yt, w, h, r, 3)
        val out = face.copyOf()
        for (i in face.indices) {
            if (m[i] == 0f || bm[i] < 0.05f) continue
            val d = ((bt[i] - bs[i]) / bm[i]).coerceIn(-60f, 60f)
            val c = face[i]
            out[i] = Pixels.rgb(c, Pixels.clamp255((c shr 16 and 0xff) + d), Pixels.clamp255((c shr 8 and 0xff) + d), Pixels.clamp255((c and 0xff) + d))
        }
        return out
    }

    /** 眼睛、嘴的张开程度：轮廓外接框的高 / 宽 */
    fun openness(pts: FloatArray, idx: IntArray): Float {
        val b = bounds(polygon(pts, idx))
        return if (b[2] <= 0f) 0f else b[3] / b[2]
    }

    /**
     * 两张脸这个部位张开程度差不多（比如都睁眼、都闭嘴）就一起换，五官更像参考脸；差得多（一个眯眼一个睁眼、
     * 一个大笑一个闭嘴）就保留目标自己的，否则会拉出两团黑或撕裂的嘴
     */
    fun similarOpenness(a: Float, b: Float): Boolean {
        if (maxOf(a, b) < 0.08f) return true // 都闭着
        return minOf(a, b) / maxOf(a, b) >= 0.75f
    }

    /**
     * 5 个关键点（图里的左眼中心、右眼中心、鼻尖、左嘴角、右嘴角），给神经网络换脸对齐用。
     * [eyeA]、[eyeB] 是两只眼睛的轮廓编号（不管哪只在左），[nose] 鼻尖，[mouthA]、[mouthB] 两个嘴角。
     */
    fun fivePoints(pts: FloatArray, eyeA: IntArray, eyeB: IntArray, nose: Int, mouthA: Int, mouthB: Int): FloatArray {
        fun center(idx: IntArray): FloatArray {
            var x = 0f; var y = 0f
            for (i in idx) { x += pts[2 * i]; y += pts[2 * i + 1] }
            return floatArrayOf(x / idx.size, y / idx.size)
        }
        val e1 = center(eyeA)
        val e2 = center(eyeB)
        val (le, re) = if (e1[0] <= e2[0]) e1 to e2 else e2 to e1
        val m1 = floatArrayOf(pts[2 * mouthA], pts[2 * mouthA + 1])
        val m2 = floatArrayOf(pts[2 * mouthB], pts[2 * mouthB + 1])
        val (lm, rm) = if (m1[0] <= m2[0]) m1 to m2 else m2 to m1
        return floatArrayOf(le[0], le[1], re[0], re[1], pts[2 * nose], pts[2 * nose + 1], lm[0], lm[1], rm[0], rm[1])
    }

    private fun polygon(pts: FloatArray, idx: IntArray) = FloatArray(idx.size * 2) { if (it % 2 == 0) pts[idx[it / 2] * 2] else pts[idx[it / 2] * 2 + 1] }

    /** minX, minY, 宽, 高 */
    private fun bounds(poly: FloatArray): FloatArray {
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE; var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (i in 0 until poly.size / 2) {
            minX = min(minX, poly[2 * i]); maxX = max(maxX, poly[2 * i])
            minY = min(minY, poly[2 * i + 1]); maxY = max(maxY, poly[2 * i + 1])
        }
        return floatArrayOf(minX, minY, maxX - minX, maxY - minY)
    }

    /**
     * 边缘色差修正（近似泊松融合）：在 [ring]（外轮廓的羽化边）上算「目标 − 贴上去的脸」的色差，用 push-pull 把它平滑地插值到整张脸，
     * 再加到贴上去的脸上。这样边缘和周围的皮肤颜色、明暗完全接上，脸中间的五官细节不变。
     */
    fun seamCorrect(face: IntArray, target: IntArray, ring: FloatArray, mask: FloatArray, w: Int, h: Int): IntArray {
        // 色差存成 128 为零点、缩小一半的 RGB，借用 Inpaint.fill 的插值
        val diff = IntArray(face.size)
        val unknown = FloatArray(face.size)
        var known = 0
        for (i in face.indices) {
            val m = ring[i]
            if (m > 0.02f && m < 0.6f) {
                val f = face[i]
                val t = target[i]
                fun d(sh: Int) = Pixels.clamp255((((t shr sh) and 0xff) - ((f shr sh) and 0xff)) / 2 + 128)
                diff[i] = Pixels.ALPHA or (d(16) shl 16) or (d(8) shl 8) or d(0)
                known++
            } else {
                unknown[i] = 1f
            }
        }
        if (known < 16) return face
        val field = Inpaint.fill(diff, w, h, unknown)
        val out = IntArray(face.size)
        for (i in face.indices) {
            if (mask[i] <= 0f) {
                out[i] = face[i]
                continue
            }
            val f = face[i]
            val c = field[i]
            fun v(sh: Int) = Pixels.clamp255(((f shr sh) and 0xff) + (((c shr sh) and 0xff) - 128) * 2)
            out[i] = Pixels.rgb(f, v(16), v(8), v(0))
        }
        return out
    }

    /** 多边形填充成 0/1 */
    fun fill(poly: FloatArray, w: Int, h: Int): FloatArray {
        val m = FloatArray(w * h)
        val n = poly.size / 2
        if (n < 3) return m
        val b = bounds(poly)
        val xs = FloatArray(n)
        for (y in max(0, b[1].toInt())..min(h - 1, (b[1] + b[3]).toInt())) {
            val py = y + 0.5f
            var k = 0
            for (i in 0 until n) {
                val ax = poly[2 * i]; val ay = poly[2 * i + 1]
                val j = (i + 1) % n
                val bx = poly[2 * j]; val by = poly[2 * j + 1]
                if ((ay <= py && by > py) || (by <= py && ay > py)) xs[k++] = ax + (py - ay) / (by - ay) * (bx - ax)
            }
            xs.sort(0, k)
            var p = 0
            while (p + 1 < k) {
                for (x in max(0, ceil(xs[p] - 0.5f).toInt())..min(w - 1, floor(xs[p + 1] - 0.5f).toInt())) m[y * w + x] = 1f
                p += 2
            }
        }
        return m
    }

    /** 逐个三角形反向映射：dst 三角形里的每个像素算出在 src 三角形里的位置，双线性取色。返回（贴好的图，覆盖 0/1） */
    fun warp(src: IntArray, sw: Int, sh: Int, srcPts: FloatArray, dw: Int, dh: Int, dstPts: FloatArray, tris: IntArray): Pair<IntArray, FloatArray> {
        val out = IntArray(dw * dh)
        val cover = FloatArray(dw * dh)
        var t = 0
        while (t + 2 < tris.size) {
            val a = tris[t]
            val b = tris[t + 1]
            val c = tris[t + 2]
            t += 3
            val x0 = dstPts[2 * a]; val y0 = dstPts[2 * a + 1]
            val x1 = dstPts[2 * b]; val y1 = dstPts[2 * b + 1]
            val x2 = dstPts[2 * c]; val y2 = dstPts[2 * c + 1]
            val den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if (abs(den) < 1e-6f) continue
            val sx0 = srcPts[2 * a]; val sy0 = srcPts[2 * a + 1]
            val sx1 = srcPts[2 * b]; val sy1 = srcPts[2 * b + 1]
            val sx2 = srcPts[2 * c]; val sy2 = srcPts[2 * c + 1]
            val minX = max(0, floor(min(x0, min(x1, x2))).toInt())
            val maxX = min(dw - 1, ceil(max(x0, max(x1, x2))).toInt())
            val minY = max(0, floor(min(y0, min(y1, y2))).toInt())
            val maxY = min(dh - 1, ceil(max(y0, max(y1, y2))).toInt())
            for (y in minY..maxY) {
                val py = y + 0.5f
                for (x in minX..maxX) {
                    val px = x + 0.5f
                    val l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                    val l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                    val l2 = 1f - l0 - l1
                    if (l0 < -0.01f || l1 < -0.01f || l2 < -0.01f) continue
                    val i = y * dw + x
                    out[i] = sample(src, sw, sh, l0 * sx0 + l1 * sx1 + l2 * sx2 - 0.5f, l0 * sy0 + l1 * sy1 + l2 * sy2 - 0.5f)
                    cover[i] = 1f
                }
            }
        }
        return out to cover
    }

    /** 双线性取色（坐标是像素左上角为原点，超出边界按边缘取），结果不透明 */
    internal fun sample(px: IntArray, w: Int, h: Int, fx: Float, fy: Float): Int {
        val x = fx.coerceIn(0f, (w - 1).toFloat())
        val y = fy.coerceIn(0f, (h - 1).toFloat())
        val x0 = x.toInt()
        val y0 = y.toInt()
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)
        val tx = x - x0
        val ty = y - y0
        val top = Pixels.lerp(px[y0 * w + x0], px[y0 * w + x1], tx)
        val bot = Pixels.lerp(px[y1 * w + x0], px[y1 * w + x1], tx)
        return Pixels.lerp(top, bot, ty) or Pixels.ALPHA
    }

    /** 轮廓多边形填充，再往里收、羽化：边缘一圈从 1 渐变到 0，不会有硬边 */
    fun faceMask(poly: FloatArray, w: Int, h: Int): FloatArray {
        val m = fill(poly, w, h)
        if (poly.size < 6) return m
        // 羽化半径按脸的大小：脸宽的 1/10
        val r = max(2, (bounds(poly)[2] / 10f).toInt())
        val blurred = Pixels.blurF(m, w, h, r, 2)
        for (i in m.indices) m[i] = Pixels.smoothstep(0.5f, 0.95f, blurred[i])
        return m
    }

    /**
     * 按蒙版内的统计把参考脸的颜色调到目标脸：在亮度 + 色度（YCbCr）里做。亮度按均值和反差对齐；色度均值对齐、
     * 饱和度按两边色度的离散程度缩放（目标是黑白、泛黄的老照片时，贴上去的脸也跟着变成同样的色调）。
     * 不在 RGB 三个通道上各自对齐：低饱和的目标会把某个通道拉偏，出现发蓝的眉毛、发黄的斑块。
     */
    fun matchColor(src: IntArray, target: IntArray, mask: FloatArray): IntArray {
        val ms = stats(src, mask)
        val mt = stats(target, mask)
        if (ms == null || mt == null) return src
        val gY = (mt[3] / max(ms[3], 1f)).coerceIn(0.7f, 1.4f)
        val chromaS = sqrt((ms[4] * ms[4] + ms[5] * ms[5]) / 2f)
        val chromaT = sqrt((mt[4] * mt[4] + mt[5] * mt[5]) / 2f)
        val gC = (chromaT / max(chromaS, 1f)).coerceIn(0f, 1.2f)
        val out = IntArray(src.size)
        for (i in src.indices) {
            if (mask[i] <= 0f) {
                out[i] = src[i]
                continue
            }
            val c = src[i]
            val r = (c shr 16 and 0xff).toFloat()
            val g = (c shr 8 and 0xff).toFloat()
            val b = (c and 0xff).toFloat()
            val y = 0.299f * r + 0.587f * g + 0.114f * b
            val y2 = (y - ms[0]) * gY + mt[0]
            val cb2 = ((b - y) / 1.772f - ms[1]) * gC + mt[1]
            val cr2 = ((r - y) / 1.402f - ms[2]) * gC + mt[2]
            out[i] = Pixels.rgb(
                c,
                Pixels.clamp255(y2 + 1.402f * cr2),
                Pixels.clamp255(y2 - 0.344136f * cb2 - 0.714136f * cr2),
                Pixels.clamp255(y2 + 1.772f * cb2),
            )
        }
        return out
    }

    /** 蒙版加权的 Y、Cb、Cr（色度以 0 为中心）均值（0..2）和标准差（3..5）；蒙版几乎为空时 null */
    private fun stats(px: IntArray, mask: FloatArray): FloatArray? {
        var wsum = 0.0
        var sy = 0.0; var sb = 0.0; var sr = 0.0
        var sy2 = 0.0; var sb2 = 0.0; var sr2 = 0.0
        for (i in px.indices) {
            val m = mask[i]
            if (m < 0.5f) continue
            val c = px[i]
            val r = (c shr 16 and 0xff).toDouble()
            val g = (c shr 8 and 0xff).toDouble()
            val b = (c and 0xff).toDouble()
            val y = 0.299 * r + 0.587 * g + 0.114 * b
            val cb = (b - y) / 1.772
            val cr = (r - y) / 1.402
            sy += y * m; sb += cb * m; sr += cr * m
            sy2 += y * y * m; sb2 += cb * cb * m; sr2 += cr * cr * m
            wsum += m
        }
        if (wsum < 16) return null
        val mean = doubleArrayOf(sy / wsum, sb / wsum, sr / wsum)
        val sq = doubleArrayOf(sy2 / wsum, sb2 / wsum, sr2 / wsum)
        return FloatArray(6) { k -> if (k < 3) mean[k].toFloat() else sqrt(max(0.0, sq[k - 3] - mean[k - 3] * mean[k - 3])).toFloat() }
    }
}
