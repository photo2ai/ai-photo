package com.photo2ai.aiphoto.core

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.ensureActive
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.coroutines.coroutineContext

/** 本地模型推理（ONNX Runtime，CPU） */
object LocalAi {
    private val env by lazy { OrtEnvironment.getEnvironment() }
    private val sessions = HashMap<String, OrtSession>()

    class BrokenModel(message: String) : Exception(message)

    /** 加载失败说明模型文件坏了：删掉，提示重试（下次会重新下载） */
    @Synchronized
    private fun session(file: File): OrtSession = sessions.getOrPut(file.path) {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        try {
            env.createSession(file.path, opts)
        } catch (e: OrtException) {
            LocalModels.discard(file)
            throw BrokenModel("模型文件损坏，已经删除，再试一次会重新下载")
        }
    }

    private fun direct(bytes: ByteArray): ByteBuffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).put(bytes).apply { rewind() }

    /**
     * MI-GAN 补图。[mask] 0..1，> 0.04 的地方要补；边缘按蒙版值和原图混合。
     * 只把涂抹范围外扩一圈（给模型看周围的背景）裁出来算，再贴回去：大图上省内存也快。
     */
    fun inpaint(model: File, px: IntArray, w: Int, h: Int, mask: FloatArray): IntArray {
        var x0 = w; var y0 = h; var x1 = -1; var y1 = -1
        for (y in 0 until h) for (x in 0 until w) if (mask[y * w + x] > 0.04f) {
            if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y
        }
        if (x1 < 0) return px.copyOf()
        val margin = maxOf(64, maxOf(x1 - x0, y1 - y0))
        val cx0 = maxOf(0, x0 - margin); val cy0 = maxOf(0, y0 - margin)
        val cw = minOf(w, x1 + margin + 1) - cx0; val ch = minOf(h, y1 + margin + 1) - cy0
        val cpx = IntArray(cw * ch)
        val cmask = FloatArray(cw * ch)
        for (y in 0 until ch) {
            System.arraycopy(px, (cy0 + y) * w + cx0, cpx, y * cw, cw)
            System.arraycopy(mask, (cy0 + y) * w + cx0, cmask, y * cw, cw)
        }
        // MI-GAN 的流水线在洞的边缘会和输入像素混合，洞里原来的颜色（要去掉的东西）会从边上透出来。
        // 先用 push-pull 把洞填成周围的背景色再送进去
        val prefilled = Inpaint.fill(cpx, cw, ch, cmask)
        val filled = inpaintWhole(model, prefilled, cw, ch, cmask)
        val out = px.copyOf()
        for (y in 0 until ch) System.arraycopy(filled, y * cw, out, (cy0 + y) * w + cx0, cw)
        return out
    }

    private fun inpaintWhole(model: File, px: IntArray, w: Int, h: Int, mask: FloatArray): IntArray {
        val s = session(model)
        val shapeImg = longArrayOf(1, 3, h.toLong(), w.toLong())
        val shapeMask = longArrayOf(1, 1, h.toLong(), w.toLong())
        OnnxTensor.createTensor(env, direct(Tensors.toChwU8(px)), shapeImg, OnnxJavaType.UINT8).use { img ->
            OnnxTensor.createTensor(env, direct(Tensors.miganMask(mask)), shapeMask, OnnxJavaType.UINT8).use { m ->
                s.run(mapOf("image" to img, "mask" to m)).use { r ->
                    val buf = (r.get(0) as OnnxTensor).byteBuffer
                    val bytes = ByteArray(buf.remaining()).also { buf.get(it) }
                    val filled = Tensors.fromChwU8(bytes, px)
                    return IntArray(px.size) { i ->
                        val v = mask[i]
                        if (v <= 0.04f) px[i] else Pixels.lerp(px[i], filled[i], minOf(1f, v * 1.5f))
                    }
                }
            }
        }
    }

    /**
     * Real-ESRGAN 放大 4 倍，分块算（默认块 192、重叠 12），每块算完报一次进度；协程取消时在块之间停下。
     */
    suspend fun upscale4x(model: File, px: IntArray, w: Int, h: Int, tile: Int = 192, onProgress: (Float) -> Unit): IntArray {
        val s = session(model)
        val out = IntArray(w * 4 * h * 4)
        val tiles = Tensors.tiles(w, h, tile, 12)
        tiles.forEachIndexed { k, t ->
            coroutineContext.ensureActive()
            val data = Tensors.tileToChwF32(px, w, t.x, t.y, t.w, t.h)
            OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(1, 3, t.h.toLong(), t.w.toLong())).use { input ->
                s.run(mapOf("input" to input)).use { r ->
                    val fb = (r.get(0) as OnnxTensor).floatBuffer
                    val arr = FloatArray(fb.remaining()).also { fb.get(it) }
                    Tensors.putScaledTile(out, w * 4, t, arr, 4, px, w)
                }
            }
            onProgress((k + 1f) / tiles.size)
        }
        return out
    }
}
