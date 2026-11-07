package com.photo2ai.aiphoto.core

import ai.onnxruntime.OnnxTensor
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 高清换脸（本地神经网络）：ArcFace 取参考脸的身份特征 → inswapper 在目标脸（对齐到 128×128）上换成这个身份 →
 * Real-ESRGAN 把 128 的脸放大到 512 变清楚 → 按蒙版贴回原图。和 insightface / FaceFusion 的流程一致。
 * 模型只允许非商业使用（InsightFace 的许可）。
 */
object NeuralSwap {
    private var emapCache: Pair<String, FloatArray>? = null

    /**
     * inswapper 的 emap（512×512 矩阵）存在模型文件里最后一个 initializer 中（FaceFusion 也是这样取的）。
     * 直接从下载好的模型文件里按 protobuf 格式找出来，不另外托管这份从模型导出的数据。
     */
    @Synchronized
    fun emapOf(model: File): FloatArray {
        emapCache?.let { if (it.first == model.path) return it.second }
        val e = OnnxInitializer.last(model, 512 * 512)
        emapCache = model.path to e
        return e
    }

    /** 参考脸的 ArcFace 特征（512 维，已归一化）。[five] 是原图里的 5 个关键点 */
    fun embedding(arcface: File, px: IntArray, w: Int, h: Int, five: FloatArray): FloatArray {
        val m = FaceAlign.similarity(five, FaceAlign.ARCFACE_112)
        val face = FaceAlign.warp(px, w, h, m, 112, 112)
        val input = FloatArray(3 * 112 * 112)
        val n = 112 * 112
        for (i in 0 until n) {
            val c = face[i]
            input[i] = ((c shr 16 and 0xff) - 127.5f) / 127.5f
            input[n + i] = ((c shr 8 and 0xff) - 127.5f) / 127.5f
            input[2 * n + i] = ((c and 0xff) - 127.5f) / 127.5f
        }
        val s = LocalAi.session(arcface)
        OnnxTensor.createTensor(LocalAi.env, FloatBuffer.wrap(input), longArrayOf(1, 3, 112, 112)).use { t ->
            s.run(mapOf(s.inputNames.first() to t)).use { r ->
                val fb = (r.get(0) as OnnxTensor).floatBuffer
                val e = FloatArray(fb.remaining()).also { fb.get(it) }
                return normalize(e)
            }
        }
    }

    private fun normalize(v: FloatArray): FloatArray {
        var s = 0.0
        for (x in v) s += x * x
        val k = (1.0 / max(sqrt(s), 1e-12)).toFloat()
        return FloatArray(v.size) { v[it] * k }
    }

    /** inswapper 要的 source：特征乘 emap 再归一化 */
    fun latent(emb: FloatArray, emap: FloatArray): FloatArray {
        val out = FloatArray(512)
        for (j in 0 until 512) {
            var s = 0f
            for (i in 0 until 512) s += emb[i] * emap[i * 512 + j]
            out[j] = s
        }
        return normalize(out)
    }

    /**
     * 在 [px] 的目标脸上换成 [emb] 的身份，返回新的整图像素。[five] 是目标脸的 5 个关键点；[region] 是原图坐标的
     * 额外蒙版（0..1，比如脸部轮廓，不限制就传 null）；[esrgan] 不为 null 时把换好的脸放大到 512 再贴回去。
     */
    suspend fun swap(
        inswapper: File, emap: FloatArray, esrgan: File?,
        px: IntArray, w: Int, h: Int, five: FloatArray, emb: FloatArray, region: FloatArray?,
    ): IntArray {
        val m128 = FaceAlign.similarity(five, FaceAlign.ARCFACE_128)
        val face = FaceAlign.warp(px, w, h, m128, 128, 128)
        val target = Tensors.tileToChwF32(face, 128, 0, 0, 128, 128)
        val s = LocalAi.session(inswapper)
        val names = s.inputNames.toList()
        val tName = names.firstOrNull { it.contains("target") } ?: names[0]
        val sName = names.firstOrNull { it.contains("source") } ?: names[1]
        var out: IntArray
        OnnxTensor.createTensor(LocalAi.env, FloatBuffer.wrap(target), longArrayOf(1, 3, 128, 128)).use { tt ->
            OnnxTensor.createTensor(LocalAi.env, FloatBuffer.wrap(latent(emb, emap)), longArrayOf(1, 512)).use { st ->
                s.run(mapOf(tName to tt, sName to st)).use { r ->
                    val fb = (r.get(0) as OnnxTensor).floatBuffer
                    val f = FloatArray(fb.remaining()).also { fb.get(it) }
                    out = IntArray(128 * 128)
                    Tensors.putScaledTile(out, 128, Tensors.Tile(0, 0, 128, 128, 0, 0, 128, 128), f, 1, face, 128)
                }
            }
        }
        // 只贴模型真正改过的地方：和输入比，颜色差得多的像素是脸，外扩、羽化成蒙版；头发、背景保留原图的清晰像素
        var changed = changedMask(face, out, 128)
        var size = 128
        var m = m128
        if (esrgan != null) {
            out = LocalAi.upscale4x(esrgan, out, 128, 128) {}
            size = 512
            m = FloatArray(6) { m128[it] * 4f }
            changed = Pixels.resizeF(changed, 128, 128, 512, 512)
        }
        return pasteBack(px, w, h, out, size, m, region, changed)
    }

    /** 模型改动的范围：颜色差超过 6 开始算、24 算满，再外扩、羽化，变成连成一片的软蒙版 */
    fun changedMask(before: IntArray, after: IntArray, size: Int): FloatArray {
        val d = FloatArray(size * size) { i ->
            val a = before[i]
            val b = after[i]
            val dist = maxOf(
                kotlin.math.abs((a shr 16 and 0xff) - (b shr 16 and 0xff)),
                kotlin.math.abs((a shr 8 and 0xff) - (b shr 8 and 0xff)),
                kotlin.math.abs((a and 0xff) - (b and 0xff)),
            )
            Pixels.smoothstep(6f, 24f, dist.toFloat())
        }
        val r = max(2, size / 40)
        val grown = Pixels.blurF(d, size, size, r, 2)
        for (i in grown.indices) grown[i] = (grown[i] * 2.5f).coerceAtMost(1f)
        return Pixels.blurF(grown, size, size, r, 2)
    }

    /**
     * 贴回：对齐空间里的方框蒙版（四周一圈渐变到 0）乘上 [changed]（对齐空间）和 [region]（原图空间），
     * 原图上每个像素反算到对齐空间取换好的脸混合进去。
     */
    fun pasteBack(px: IntArray, w: Int, h: Int, face: IntArray, size: Int, m: FloatArray, region: FloatArray?, changed: FloatArray? = null): IntArray {
        val box = FloatArray(size * size)
        val edge = max(2, (size * 0.08f).toInt())
        for (y in edge until size - edge) for (x in edge until size - edge) box[y * size + x] = 1f
        val soft = Pixels.blurF(box, size, size, max(1, (size * 0.04f).toInt()), 3)
        val out = px.copyOf()
        val b = FaceAlign.backBounds(m, size, w, h)
        for (y in b[1] until b[3]) for (x in b[0] until b[2]) {
            val p = FaceAlign.apply(m, x + 0.5f, y + 0.5f)
            val fx = p[0] - 0.5f
            val fy = p[1] - 0.5f
            if (fx < 0 || fy < 0 || fx > size - 1 || fy > size - 1) continue
            val i = y * w + x
            var k = sampleF(soft, size, fx, fy)
            if (changed != null) k *= sampleF(changed, size, fx, fy)
            if (region != null) k *= region[i]
            if (k <= 0f) continue
            out[i] = Pixels.lerp(px[i], FaceAlign.sample(face, size, size, fx, fy) or Pixels.ALPHA, k.coerceAtMost(1f))
        }
        return out
    }

    private fun sampleF(a: FloatArray, size: Int, fx: Float, fy: Float): Float {
        val x0 = fx.toInt().coerceIn(0, size - 1)
        val y0 = fy.toInt().coerceIn(0, size - 1)
        val x1 = (x0 + 1).coerceAtMost(size - 1)
        val y1 = (y0 + 1).coerceAtMost(size - 1)
        val tx = fx - x0
        val ty = fy - y0
        val top = a[y0 * size + x0] * (1 - tx) + a[y0 * size + x1] * tx
        val bot = a[y1 * size + x0] * (1 - tx) + a[y1 * size + x1] * tx
        return top * (1 - ty) + bot * ty
    }
}
