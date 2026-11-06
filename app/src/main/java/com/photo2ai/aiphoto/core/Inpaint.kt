package com.photo2ai.aiphoto.core

import kotlin.math.max
import kotlin.math.min

/**
 * 本机快速消除：push-pull 金字塔填充。把涂抹区域当成未知像素，逐级缩小时只用已知像素求平均，再逐级放大把空洞补上，
 * 得到从四周平滑过渡的颜色。适合去掉小瑕疵、电线、小物件；大面积、有纹理的区域交给云端 AI 消除。
 */
object Inpaint {
    /**
     * [mask] 每个像素 0..1，> 0.04 的算空洞要填；边缘按蒙版值和原图混合，没有硬边。返回新数组。
     */
    fun fill(px: IntArray, w: Int, h: Int, mask: FloatArray): IntArray {
        var x0 = w
        var y0 = h
        var x1 = -1
        var y1 = -1
        for (y in 0 until h) for (x in 0 until w) {
            if (mask[y * w + x] > HOLE) {
                if (x < x0) x0 = x
                if (x > x1) x1 = x
                if (y < y0) y0 = y
                if (y > y1) y1 = y
            }
        }
        if (x1 < 0) return px.copyOf()
        // 只在空洞外扩一圈的范围里算，大图上涂一小块也很快
        val margin = max(8, max(x1 - x0, y1 - y0) / 2)
        val rx0 = max(0, x0 - margin)
        val ry0 = max(0, y0 - margin)
        val rw = min(w - 1, x1 + margin) - rx0 + 1
        val rh = min(h - 1, y1 + margin) - ry0 + 1

        val levels = ArrayList<Level>()
        var lv = Level(rw, rh)
        var known = 0
        for (y in 0 until rh) for (x in 0 until rw) {
            val idx = (ry0 + y) * w + rx0 + x
            if (mask[idx] <= HOLE) {
                val c = px[idx]
                val j = y * rw + x
                lv.r[j] = (c shr 16 and 0xff).toFloat()
                lv.g[j] = (c shr 8 and 0xff).toFloat()
                lv.b[j] = (c and 0xff).toFloat()
                lv.wt[j] = 1f
                known++
            }
        }
        if (known == 0) return px.copyOf() // 整张图都涂了：没有可以参照的像素
        levels += lv

        // 缩小：每个粗像素是 2x2 子像素按权重的平均，权重封顶 1
        while (lv.w > 1 || lv.h > 1) {
            val nw = (lv.w + 1) / 2
            val nh = (lv.h + 1) / 2
            val c = Level(nw, nh)
            for (y in 0 until nh) for (x in 0 until nw) {
                var sw = 0f
                var sr = 0f
                var sg = 0f
                var sb = 0f
                for (dy in 0..1) for (dx in 0..1) {
                    val fx = 2 * x + dx
                    val fy = 2 * y + dy
                    if (fx >= lv.w || fy >= lv.h) continue
                    val j = fy * lv.w + fx
                    val wt = lv.wt[j]
                    sw += wt; sr += lv.r[j] * wt; sg += lv.g[j] * wt; sb += lv.b[j] * wt
                }
                val j = y * nw + x
                if (sw > 0f) {
                    c.r[j] = sr / sw; c.g[j] = sg / sw; c.b[j] = sb / sw
                }
                c.wt[j] = min(1f, sw)
            }
            levels += c
            lv = c
        }

        // 放大：权重不足 1 的像素用上一级双线性插值的颜色补齐
        for (l in levels.size - 2 downTo 0) {
            val f = levels[l]
            val c = levels[l + 1]
            for (y in 0 until f.h) {
                val sy = ((y + 0.5f) / 2f - 0.5f).coerceIn(0f, (c.h - 1).toFloat())
                val cy0 = sy.toInt()
                val cy1 = min(cy0 + 1, c.h - 1)
                val ty = sy - cy0
                for (x in 0 until f.w) {
                    val j = y * f.w + x
                    val wt = f.wt[j]
                    if (wt >= 1f) continue
                    val sx = ((x + 0.5f) / 2f - 0.5f).coerceIn(0f, (c.w - 1).toFloat())
                    val cx0 = sx.toInt()
                    val cx1 = min(cx0 + 1, c.w - 1)
                    val tx = sx - cx0
                    val a = cy0 * c.w + cx0
                    val b = cy0 * c.w + cx1
                    val d = cy1 * c.w + cx0
                    val e = cy1 * c.w + cx1
                    val w00 = (1 - tx) * (1 - ty)
                    val w10 = tx * (1 - ty)
                    val w01 = (1 - tx) * ty
                    val w11 = tx * ty
                    val ur = c.r[a] * w00 + c.r[b] * w10 + c.r[d] * w01 + c.r[e] * w11
                    val ug = c.g[a] * w00 + c.g[b] * w10 + c.g[d] * w01 + c.g[e] * w11
                    val ub = c.b[a] * w00 + c.b[b] * w10 + c.b[d] * w01 + c.b[e] * w11
                    f.r[j] = f.r[j] * wt + ur * (1 - wt)
                    f.g[j] = f.g[j] * wt + ug * (1 - wt)
                    f.b[j] = f.b[j] * wt + ub * (1 - wt)
                    f.wt[j] = 1f
                }
            }
        }

        val out = px.copyOf()
        val base = levels[0]
        for (y in 0 until rh) for (x in 0 until rw) {
            val idx = (ry0 + y) * w + rx0 + x
            val m = mask[idx]
            if (m <= HOLE) continue
            val j = y * rw + x
            val filled = Pixels.rgb(px[idx], Pixels.clamp255(base.r[j]), Pixels.clamp255(base.g[j]), Pixels.clamp255(base.b[j]))
            out[idx] = Pixels.lerp(px[idx], filled, min(1f, m * 1.5f))
        }
        return out
    }

    private const val HOLE = 0.04f

    private class Level(val w: Int, val h: Int) {
        val r = FloatArray(w * h)
        val g = FloatArray(w * h)
        val b = FloatArray(w * h)
        val wt = FloatArray(w * h)
    }
}
