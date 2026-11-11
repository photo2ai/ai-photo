package com.photo2ai.aiphoto.core

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** 二维向量（像素坐标），本地换装的几何计算用 */
data class V(val x: Float, val y: Float) {
    operator fun plus(o: V) = V(x + o.x, y + o.y)
    operator fun minus(o: V) = V(x - o.x, y - o.y)
    operator fun times(k: Float) = V(x * k, y * k)
    operator fun div(k: Float) = V(x / k, y / k)
    operator fun unaryMinus() = V(-x, -y)
    infix fun dot(o: V) = x * o.x + y * o.y
    fun norm() = hypot(x, y)
    fun unit(): V = this / (norm() + 1e-6f)

    /** 逆时针转 90°（图像坐标里 y 朝下，和 numpy 版的 rot90 一致：(-y, x)） */
    fun rot90() = V(-y, x)
}

/**
 * 纯 Kotlin 的二值蒙版运算（BooleanArray，行优先 w×h），能在 JVM 上单测。
 * 膨胀 / 腐蚀用正方形结构元素（numpy 原型里是十字形迭代，效果差不多）。
 */
object Masks {
    fun count(m: BooleanArray): Int = m.count { it }

    /** 正方形半径 r 的膨胀：横竖各一遍「窗口里有没有 true」 */
    fun dilate(m: BooleanArray, w: Int, h: Int, r: Int): BooleanArray {
        if (r <= 0) return m.copyOf()
        val tmp = BooleanArray(m.size)
        val out = BooleanArray(m.size)
        val pre = IntArray(max(w, h) + 1)
        for (y in 0 until h) {
            for (x in 0 until w) pre[x + 1] = pre[x] + if (m[y * w + x]) 1 else 0
            for (x in 0 until w) tmp[y * w + x] = pre[min(w, x + r + 1)] - pre[max(0, x - r)] > 0
        }
        for (x in 0 until w) {
            for (y in 0 until h) pre[y + 1] = pre[y] + if (tmp[y * w + x]) 1 else 0
            for (y in 0 until h) out[y * w + x] = pre[min(h, y + r + 1)] - pre[max(0, y - r)] > 0
        }
        return out
    }

    /** 腐蚀 = 对补集膨胀再取反（图像边界外当作 false，和 scipy 默认一致） */
    fun erode(m: BooleanArray, w: Int, h: Int, r: Int): BooleanArray {
        if (r <= 0) return m.copyOf()
        // 外面补一圈 false，边缘也会被腐蚀
        val pw = w + 2 * r
        val ph = h + 2 * r
        val inv = BooleanArray(pw * ph) { true }
        for (y in 0 until h) for (x in 0 until w) inv[(y + r) * pw + x + r] = !m[y * w + x]
        val d = dilate(inv, pw, ph, r)
        return BooleanArray(m.size) { !d[(it / w + r) * pw + it % w + r] }
    }

    fun open(m: BooleanArray, w: Int, h: Int, r: Int) = dilate(erode(m, w, h, r), w, h, r)
    fun close(m: BooleanArray, w: Int, h: Int, r: Int) = erode(dilate(m, w, h, r), w, h, r)

    /** 4 连通标号：返回 (标号数组（0 是背景，从 1 起）, 各块大小（下标 = 标号 - 1）) */
    fun label(m: BooleanArray, w: Int, h: Int): Pair<IntArray, IntArray> {
        val lab = IntArray(m.size)
        val sizes = ArrayList<Int>()
        val queue = IntArray(m.size)
        for (start in m.indices) {
            if (!m[start] || lab[start] != 0) continue
            val id = sizes.size + 1
            var head = 0
            var tail = 0
            queue[tail++] = start
            lab[start] = id
            while (head < tail) {
                val p = queue[head++]
                val x = p % w
                val y = p / w
                if (x > 0 && m[p - 1] && lab[p - 1] == 0) { lab[p - 1] = id; queue[tail++] = p - 1 }
                if (x < w - 1 && m[p + 1] && lab[p + 1] == 0) { lab[p + 1] = id; queue[tail++] = p + 1 }
                if (y > 0 && m[p - w] && lab[p - w] == 0) { lab[p - w] = id; queue[tail++] = p - w }
                if (y < h - 1 && m[p + w] && lab[p + w] == 0) { lab[p + w] = id; queue[tail++] = p + w }
            }
            sizes += tail
        }
        return lab to sizes.toIntArray()
    }

    /** 只留最大的一块 */
    fun largest(m: BooleanArray, w: Int, h: Int): BooleanArray {
        val (lab, sizes) = label(m, w, h)
        if (sizes.isEmpty()) return m.copyOf()
        val best = sizes.indices.maxBy { sizes[it] } + 1
        return BooleanArray(m.size) { lab[it] == best }
    }

    /** 去掉小于 [minSize] 像素的块 */
    fun dropSmall(m: BooleanArray, w: Int, h: Int, minSize: Int): BooleanArray {
        val (lab, sizes) = label(m, w, h)
        return BooleanArray(m.size) { lab[it] != 0 && sizes[lab[it] - 1] >= minSize }
    }

    /** 填洞：从图像边缘出发够不到的背景都算洞 */
    fun fillHoles(m: BooleanArray, w: Int, h: Int): BooleanArray {
        val outside = BooleanArray(m.size)
        val queue = IntArray(m.size)
        var tail = 0
        fun push(p: Int) {
            if (!m[p] && !outside[p]) { outside[p] = true; queue[tail++] = p }
        }
        for (x in 0 until w) { push(x); push((h - 1) * w + x) }
        for (y in 0 until h) { push(y * w); push(y * w + w - 1) }
        var head = 0
        while (head < tail) {
            val p = queue[head++]
            val x = p % w
            val y = p / w
            if (x > 0) push(p - 1)
            if (x < w - 1) push(p + 1)
            if (y > 0) push(p - w)
            if (y < h - 1) push(p + w)
        }
        return BooleanArray(m.size) { !outside[it] }
    }

    /** 折线 [pts] 周围半径 [r] 以内的像素（「胶囊」：手臂、手的范围），结果并到 [into] 上 */
    fun capsule(w: Int, h: Int, pts: List<V>, r: Float, into: BooleanArray = BooleanArray(w * h)): BooleanArray {
        val r2 = r * r
        val segs = if (pts.size == 1) listOf(pts[0] to pts[0]) else pts.zipWithNext()
        for ((a, b) in segs) {
            val x0 = max(0, floor(min(a.x, b.x) - r).toInt())
            val x1 = min(w - 1, ceil(max(a.x, b.x) + r).toInt())
            val y0 = max(0, floor(min(a.y, b.y) - r).toInt())
            val y1 = min(h - 1, ceil(max(a.y, b.y) + r).toInt())
            val d = b - a
            val len2 = d dot d
            for (y in y0..y1) for (x in x0..x1) {
                val px = x - a.x
                val py = y - a.y
                val t = if (len2 > 0f) ((px * d.x + py * d.y) / len2).coerceIn(0f, 1f) else 0f
                val dx = px - t * d.x
                val dy = py - t * d.y
                if (dx * dx + dy * dy <= r2) into[y * w + x] = true
            }
        }
        return into
    }

    /** 凸包（单调链），逆时针 */
    fun convexHull(points: List<V>): List<V> {
        val p = points.distinct().sortedWith(compareBy({ it.x }, { it.y }))
        if (p.size < 3) return p
        fun cross(o: V, a: V, b: V) = (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        val lower = ArrayList<V>()
        for (q in p) {
            while (lower.size >= 2 && cross(lower[lower.size - 2], lower.last(), q) <= 0) lower.removeAt(lower.size - 1)
            lower += q
        }
        val upper = ArrayList<V>()
        for (q in p.asReversed()) {
            while (upper.size >= 2 && cross(upper[upper.size - 2], upper.last(), q) <= 0) upper.removeAt(upper.size - 1)
            upper += q
        }
        return lower.dropLast(1) + upper.dropLast(1)
    }

    /** 填凸多边形（扫描线：每行取各边交点的最左、最右），结果并到 [into] 上 */
    fun fillConvex(w: Int, h: Int, poly: List<V>, into: BooleanArray = BooleanArray(w * h)): BooleanArray {
        if (poly.size < 3) return into
        val y0 = max(0, floor(poly.minOf { it.y }).toInt())
        val y1 = min(h - 1, ceil(poly.maxOf { it.y }).toInt())
        for (y in y0..y1) {
            val fy = y + 0.5f
            var lo = Float.MAX_VALUE
            var hi = -Float.MAX_VALUE
            for (i in poly.indices) {
                val a = poly[i]
                val b = poly[(i + 1) % poly.size]
                if ((a.y <= fy && b.y >= fy) || (b.y <= fy && a.y >= fy)) {
                    val x = if (abs(b.y - a.y) < 1e-6f) min(a.x, b.x) else a.x + (fy - a.y) / (b.y - a.y) * (b.x - a.x)
                    val x2 = if (abs(b.y - a.y) < 1e-6f) max(a.x, b.x) else x
                    lo = min(lo, x); hi = max(hi, x2)
                }
            }
            if (lo > hi) continue
            for (x in max(0, ceil(lo - 0.5f).toInt())..min(w - 1, floor(hi - 0.5f).toInt())) into[y * w + x] = true
        }
        return into
    }

    /** 浮点数组的中位数（只看 [sel] 为 true 的位置） */
    fun median(v: FloatArray, sel: BooleanArray): Float {
        val a = FloatArray(count(sel))
        var k = 0
        for (i in v.indices) if (sel[i]) a[k++] = v[i]
        if (k == 0) return Float.NaN
        a.sort()
        return a[k / 2]
    }

    /** 分位数（q 是 0..100，线性插值，和 numpy.percentile 默认一致） */
    fun percentile(values: FloatArray, q: Float): Float {
        if (values.isEmpty()) return Float.NaN
        val a = values.sortedArray()
        val pos = (a.size - 1) * q / 100f
        val i = floor(pos).toInt()
        val j = min(i + 1, a.size - 1)
        return a[i] + (a[j] - a[i]) * (pos - i)
    }

    fun gaussianSigmaToBox(sigma: Float): Int = max(1, (sqrt(sigma * sigma + 0.25f) - 0.5f + 0.5f).toInt())
}
