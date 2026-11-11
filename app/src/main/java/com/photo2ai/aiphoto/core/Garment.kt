package com.photo2ai.aiphoto.core

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 本地换装的衣服部分（纯 Kotlin，能单测）：从平铺 / 挂拍的衣服照片里抠出衣服，拆成躯干和左右袖子，给出肩、腋下、下摆几个点。
 * 电脑上的原型和对比见 vton-eval 的 tools/sticker_tryon2.py（parse_garment），这里逐行对照移植。
 */
object Garment {
    class Sleeve(val mask: BooleanArray, val root: V, val axis: V, val length: Float)

    class Parsed(
        val w: Int, val h: Int,
        val mask: BooleanArray, val torso: BooleanArray,
        /** 衣服最上一行、躯干下摆那一行、衣服最下一行（系带等可能比下摆还低） */
        val top: Int, val hem: Int, val bottom: Int, val cx: Int,
        val sl: V, val sr: V, val al: V, val ar: V, val hl: V, val hr: V, val neck: V,
        /** 'L'（画面左边）、'R' 两只袖子；无袖衣服为空 */
        val sleeves: Map<Char, Sleeve>,
    )

    class Bad(message: String) : Exception(message)

    /**
     * 衣服蒙版：背景色取四条边的中位色，和它差得多的算衣服；去掉零碎、只留最大一块、填洞（领口开口和外面连着，不会被填上）
     */
    fun cutout(px: IntArray, w: Int, h: Int): BooleanArray {
        val border = ArrayList<Int>()
        for (x in 0 until w) { border += px[x]; border += px[(h - 1) * w + x] }
        for (y in 0 until h) { border += px[y * w]; border += px[y * w + w - 1] }
        fun med(sh: Int) = border.map { it shr sh and 0xff }.sorted()[border.size / 2]
        // 透明背景的 PNG（商品图常见）：透明像素读出来是黑的，按颜色抠会把深色衣服当背景，直接用透明度
        val transparent = border.count { (it ushr 24) < 128 } > border.size / 2
        val br = med(16)
        val bg = med(8)
        val bb = med(0)
        var m = BooleanArray(px.size) {
            val c = px[it]
            if (transparent) (c ushr 24) >= 128
            else max(abs((c shr 16 and 0xff) - br), max(abs((c shr 8 and 0xff) - bg), abs((c and 0xff) - bb))) > 25
        }
        m = Masks.open(m, w, h, 2)
        m = Masks.largest(m, w, h)
        m = Masks.fillHoles(m, w, h)
        val cover = Masks.count(m).toFloat() / m.size
        if (cover < 0.05f || cover > 0.92f) throw Bad("没能从照片里认出衣服。请用平铺或挂起来拍、背景干净的衣服照片")
        return m
    }

    /** 这一行里包含 cx 的连续段；cx 不在衣服上（V 领开口）时取整行范围 */
    private fun centerRun(m: BooleanArray, w: Int, y: Int, cx: Int): IntArray? {
        val row = y * w
        var first = -1
        var last = -1
        for (x in 0 until w) if (m[row + x]) { if (first < 0) first = x; last = x }
        if (first < 0) return null
        if (!m[row + cx]) return intArrayOf(first, last)
        var l = cx
        while (l > 0 && m[row + l - 1]) l--
        var r = cx
        while (r < w - 1 && m[row + r + 1]) r++
        return intArrayOf(l, r)
    }

    fun parse(mask: BooleanArray, w: Int, h: Int): Parsed {
        val rows = (0 until h).filter { y -> (0 until w).any { mask[y * w + it] } }
        if (rows.size < 10) throw Bad("衣服太小了，换一张衣服占满画面的照片")
        val top = rows.first()
        val bot = rows.last()
        val hh = (bot - top).toFloat()
        // 上半部分有两只袖子，左右对称，用它定中线
        var sx = 0L
        var sn = 0
        for (y in top until (top + 0.4f * hh).toInt()) for (x in 0 until w) if (mask[y * w + x]) { sx += x; sn++ }
        val cx = (sx.toFloat() / max(sn, 1)).roundToInt().coerceIn(0, w - 1)
        // 躯干下摆 = 中线附近最低的衣服像素
        var tb = top
        for (y in top..bot) for (x in max(cx - 3, 0)..min(cx + 3, w - 1)) if (mask[y * w + x]) tb = y
        val th = (tb - top).toFloat()
        if (th < 10) throw Bad("没认出衣服的形状，换一张正面平铺的上衣照片")
        val runs = HashMap<Int, IntArray>()
        for (y in top..tb) centerRun(mask, w, y, cx)?.let { runs[y] = it }
        val lowRows = ((tb - 0.35f * th).toInt() until (tb - 0.03f * th).toInt()).filter { runs[it] != null }
        if (lowRows.isEmpty()) throw Bad("没认出衣服的形状，换一张正面平铺的上衣照片")
        val lefts = FloatArray(lowRows.size) { runs[lowRows[it]]!![0].toFloat() }
        val rights = FloatArray(lowRows.size) { runs[lowRows[it]]!![1].toFloat() }

        // 下摆以下左右各有一块 = 垂下来的长袖袖口
        var hanging: Pair<Float, Float>? = null
        if (tb + 3 < h) {
            val sub = BooleanArray(w * (h - tb - 3)) { mask[(tb + 3) * w + it] }
            val (lab, sizes) = Masks.label(sub, w, h - tb - 3)
            class Blob(var sumX: Long = 0, var n: Int = 0, var minX: Int = w, var maxX: Int = -1, var minY: Int = h, var maxY: Int = -1)
            val blobs = Array(sizes.size) { Blob() }
            for (i in lab.indices) if (lab[i] > 0) {
                val b = blobs[lab[i] - 1]
                val x = i % w
                val y = i / w
                b.sumX += x; b.n++; b.minX = min(b.minX, x); b.maxX = max(b.maxX, x); b.minY = min(b.minY, y); b.maxY = max(b.maxY, y)
            }
            val total = Masks.count(mask)
            val big = blobs.filter { it.n > 0.002f * total && it.maxY - it.minY > 0.02f * hh }
            val lb = big.filter { it.sumX.toFloat() / it.n < cx }
            val rb = big.filter { it.sumX.toFloat() / it.n > cx }
            if (lb.isNotEmpty() && rb.isNotEmpty()) hanging = lb.maxOf { it.maxX }.toFloat() to rb.minOf { it.minX }.toFloat()
        }

        var xlH: Float
        var xrH: Float
        var xlA: Float
        var xrA: Float
        var ya: Float? = null
        if (hanging != null) {
            xlH = hanging.first + 2; xrH = hanging.second - 2
            xlA = xlH; xrA = xrH
        } else {
            val wt = Masks.percentile(FloatArray(lefts.size) { rights[it] - lefts[it] }, 50f)
            for (y in (tb - 0.03f * th).toInt() downTo top + 1) {
                val r = runs[y] ?: continue
                if (r[1] - r[0] > 1.25f * wt) { ya = (y + 1).toFloat(); break }
            }
            // 躯干边线取内侧的分位数，忽略系带、蝴蝶结这类往外凸的东西
            xlH = Masks.percentile(lefts, 75f)
            xrH = Masks.percentile(rights, 25f)
            val near = if (ya != null) ((ya + 2).toInt() until (ya + 0.15f * th).toInt()).filter { runs[it] != null } else emptyList()
            if (near.isNotEmpty()) {
                xlA = Masks.percentile(FloatArray(near.size) { runs[near[it]]!![0].toFloat() }, 75f)
                xrA = Masks.percentile(FloatArray(near.size) { runs[near[it]]!![1].toFloat() }, 25f)
            } else {
                xlA = xlH; xrA = xrH
            }
        }
        // 躯干左右基本对称：一边明显更宽（系带、下摆开衩）时两边都按窄的那边
        fun sym(l: Float, r: Float): Pair<Float, Float> {
            val hl = cx - l
            val hr = r - cx
            if (abs(hl - hr) > 0.08f * (hl + hr)) {
                val m = min(hl, hr)
                return cx - m to cx + m
            }
            return l to r
        }
        sym(xlH, xrH).let { xlH = it.first; xrH = it.second }
        sym(xlA, xrA).let { xlA = it.first; xrA = it.second }
        val hasSleeves = hanging != null || (ya != null && ya > top + 0.08f * hh)
        fun topmost(x: Float): Float {
            val c = x.roundToInt().coerceIn(0, w - 1)
            for (y in 0 until h) if (mask[y * w + c]) return y.toFloat()
            return top.toFloat()
        }
        val ysl = topmost(xlA)
        val ysr = topmost(xrA)
        // 看不出腋下（垂袖、无袖）：按常见比例，肩下方 0.45 个肩宽
        val yA = if (hanging != null || !hasSleeves || ya == null) (ysl + ysr) / 2 + 0.45f * (xrA - xlA) else ya

        val sleeveZone = BooleanArray(mask.size)
        if (hasSleeves) {
            for (y in 0 until h) {
                val t = ((y - yA) / max(tb - yA, 1f)).coerceIn(0f, 1f)
                val xl = xlA + (xlH - xlA) * t
                val xr = xrA + (xrH - xrA) * t
                if (hanging == null && y >= yA + 0.04f * hh) continue
                for (x in 0 until w) {
                    val i = y * w + x
                    if (mask[i] && (x < xl - 1 || x > xr + 1)) sleeveZone[i] = true
                }
            }
        }
        val torso = BooleanArray(mask.size) { mask[it] && !sleeveZone[it] }
        val sl = V(xlA, ysl)
        val sr = V(xrA, ysr)
        val al = V(xlA, yA)
        val ar = V(xrA, yA)
        val total = Masks.count(mask)
        val sleeves = HashMap<Char, Sleeve>()
        for (side in listOf('L', 'R')) {
            var sm = BooleanArray(mask.size) { sleeveZone[it] && (if (side == 'L') it % w < cx else it % w >= cx) }
            if (Masks.count(sm) < 0.01f * total) continue
            sm = Masks.largest(sm, w, h)
            val s = if (side == 'L') sl else sr
            val a = if (side == 'L') al else ar
            val root = (s + a) / 2f
            var dmax = 0f
            for (i in sm.indices) if (sm[i]) dmax = max(dmax, (V((i % w).toFloat(), (i / w).toFloat()) - root).norm())
            // 袖口中心 = 离袖根最远的那一圈
            var cxs = 0f
            var cys = 0f
            var cn = 0
            for (i in sm.indices) if (sm[i]) {
                val p = V((i % w).toFloat(), (i / w).toFloat())
                if ((p - root).norm() > 0.85f * dmax) { cxs += p.x; cys += p.y; cn++ }
            }
            val cuff = V(cxs / max(cn, 1), cys / max(cn, 1))
            sleeves[side] = Sleeve(sm, root, (cuff - root).unit(), (cuff - root).norm())
        }
        return Parsed(
            w, h, mask, torso, top, tb, bot, cx,
            sl, sr, al, ar, V(xlH, tb.toFloat()), V(xrH, tb.toFloat()), V(cx.toFloat(), top.toFloat()), sleeves,
        )
    }
}
