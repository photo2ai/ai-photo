package com.photo2ai.aiphoto.core

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.abs
import kotlin.math.sin

/**
 * 本机 AI 加速：每个模型在这台手机上分别用 CPU、XNNPACK（ARM 优化的 CPU 计算）、NNAPI（安卓的 AI 芯片接口，
 * 全精度 / 半精度）各跑几次，结果和 CPU 一致、又最快的那个记下来，之后都用它。
 * 不同手机的 AI 芯片驱动差别很大：有的快几倍，有的更慢，少数会算错，所以必须在本机实测，不能一刀切。
 */
object Accel {
    enum class Backend(val label: String) {
        CPU("CPU"),
        XNNPACK("CPU 优化"),
        NNAPI("AI 芯片"),
        NNAPI_FP16("AI 芯片（半精度）"),
    }

    /** 变清晰分块的固定大小（块 192 + 两边各 12 重叠）：NNAPI 只能编译固定尺寸 */
    const val ESRGAN_TILE = 192
    const val ESRGAN_PAD = ESRGAN_TILE + 2 * 12

    /**
     * 每个模型的测速方式。[fixed] 是 NNAPI 要的固定尺寸（动态维度名 → 值）；[nnapi] 为 false 的模型输入尺寸每次都不同，
     * 用不了 NNAPI；[runsPerUse] 是用一次大概要跑几遍（变清晰按块算，一张图几十块），用来算总耗时。
     */
    private class Spec(val fixed: Map<String, Long>, val nnapi: Boolean, val runsPerUse: Int, val inputs: List<Pair<String, LongArray>>)

    private fun spec(m: LocalModels.Model): Spec? = when (m) {
        LocalModels.ESRGAN -> Spec(
            mapOf("height" to ESRGAN_PAD.toLong(), "width" to ESRGAN_PAD.toLong()), true, 30,
            listOf("input" to longArrayOf(1, 3, ESRGAN_PAD.toLong(), ESRGAN_PAD.toLong())),
        )
        LocalModels.ARCFACE -> Spec(mapOf("None" to 1L), true, 1, listOf("input" to longArrayOf(1, 3, 112, 112)))
        LocalModels.MIGAN -> Spec(emptyMap(), false, 1, listOf("image" to longArrayOf(1, 3, 512, 512), "mask" to longArrayOf(1, 1, 512, 512)))
        // GFPGAN 不测、固定用 CPU：卷积权重是按风格向量现算的（不是常量），NNAPI、XNNPACK 都帮不上；它一个会话跑起来
        // 就要 800 MB，测速要先后建好几个，2.5 GB 内存的模拟器上直接被系统杀掉。
        // HyperSwap 也不测、固定用 CPU：2 亿参数，测速要先后载入好几遍；第一次换脸时测完紧接着跑，释放的内存来不及还给系统，
        // 2.5 GB 的模拟器上 App 会被杀（1.0.11 测 inswapper 时踩过）
        else -> null
    }

    private fun modelOf(file: File) = LocalModels.ALL.firstOrNull { it.file == file.name }

    /** 这个模型文件该用哪种方式；没测过、关了加速、或者不认识的模型都用 CPU */
    fun chosen(file: File): Backend {
        if (!Store.accelEnabled) return Backend.CPU
        val m = modelOf(file) ?: return Backend.CPU
        return Store.accelChoice(m.sha256)?.let { name -> Backend.entries.firstOrNull { it.name == name } } ?: Backend.CPU
    }

    /** 测过没有（关了加速就当测过，不再测） */
    fun measured(m: LocalModels.Model) = !Store.accelEnabled || spec(m) == null || Store.accelChoice(m.sha256) != null

    /** 这个模型要不要测速（不测的固定用 CPU） */
    fun benchmarks(m: LocalModels.Model) = spec(m) != null

    fun options(file: File, backend: Backend): OrtSession.SessionOptions {
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        return OrtSession.SessionOptions().apply {
            // GFPGAN 是半精度存的：图优化会把权重提前展开成一份 fp32 常驻内存（电脑上实测会话 849 → 362 MB，跑一次的峰值
            // 1059 → 801 MB），关掉后每次运行时现转，速度不变
            setOptimizationLevel(
                if (modelOf(file) == LocalModels.GFPGAN) OrtSession.SessionOptions.OptLevel.NO_OPT else OrtSession.SessionOptions.OptLevel.ALL_OPT,
            )
            // HyperSwap 的中间结果很大（256×256 上上千个通道）：内存复用规划会按整张图一次性预留，电脑上实测跑一次的峰值
            // 1093 → 705 MB，速度不变
            if (modelOf(file) == LocalModels.HYPERSWAP) setMemoryPatternOptimization(false)
            when (backend) {
                Backend.CPU -> setIntraOpNumThreads(threads)
                Backend.XNNPACK -> {
                    // XNNPACK 自己管线程，ORT 这边用 1 个线程，避免两边抢 CPU
                    setIntraOpNumThreads(1)
                    addConfigEntry("session.intra_op.allow_spinning", "0")
                    addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                }
                Backend.NNAPI, Backend.NNAPI_FP16 -> {
                    setIntraOpNumThreads(threads)
                    modelOf(file)?.let { spec(it) }?.fixed?.forEach { (k, v) -> setSymbolicDimensionValue(k, v) }
                    addNnapi(if (backend == Backend.NNAPI_FP16) EnumSet.of(NNAPIFlags.USE_FP16) else EnumSet.noneOf(NNAPIFlags::class.java))
                }
            }
        }
    }

    /** 模型用加速方式加载失败（驱动不支持等）：改回 CPU 并记下，下次不再试 */
    fun fallBack(file: File) {
        val m = modelOf(file) ?: return
        Store.setAccel(m.sha256, Backend.CPU.name, Store.accelSummary(m.sha256) + "（加载失败，已改用 CPU）")
    }

    /**
     * 实测一个模型，记下最快且结果和 CPU 一致的方式，返回给用户看的一句话。
     * 每种方式：建会话的时间 + 预热一次 + 计时两次取平均。
     */
    fun benchmark(m: LocalModels.Model, file: File): String {
        val s = spec(m) ?: return ""
        val candidates = listOf(Backend.CPU, Backend.XNNPACK) + if (s.nnapi) listOf(Backend.NNAPI, Backend.NNAPI_FP16) else emptyList()
        var reference: FloatArray? = null
        val runMs = LinkedHashMap<Backend, Double>()
        val score = LinkedHashMap<Backend, Double>()
        val notes = ArrayList<String>()
        for (b in candidates) {
            try {
                val t0 = System.nanoTime()
                LocalAi.env.createSession(file.path, options(file, b)).use { session ->
                    val createMs = (System.nanoTime() - t0) / 1e6
                    val out = run(session, s)
                    val t1 = System.nanoTime()
                    repeat(2) { run(session, s) }
                    val per = (System.nanoTime() - t1) / 1e6 / 2
                    if (b == Backend.CPU) {
                        reference = out
                    } else if (!AccelMath.agrees(reference!!, out)) {
                        notes += "${b.label}结果不一致，未采用"
                        return@use
                    }
                    runMs[b] = per
                    score[b] = createMs + per * s.runsPerUse
                }
            } catch (e: Exception) {
                if (b == Backend.CPU) throw e
                notes += "${b.label}不可用"
            }
        }
        val best = AccelMath.pick(score)
        val cpu = runMs[Backend.CPU] ?: 0.0
        val unit = if (s.runsPerUse > 1) "/块" else ""
        val summary = buildString {
            append(m.title).append("：").append(best.label)
            if (best != Backend.CPU) append(" ").append(fmt(runMs[best]!!)).append(unit).append("（CPU ").append(fmt(cpu)).append(unit).append("）")
            else append(" ").append(fmt(cpu)).append(unit)
            if (notes.isNotEmpty()) append("；").append(notes.joinToString("；"))
        }
        Store.setAccel(m.sha256, best.name, summary)
        return summary
    }

    private fun fmt(ms: Double) = if (ms >= 1000) "%.1f 秒".format(ms / 1000) else "%.0f 毫秒".format(ms)

    /** 用固定的合成输入跑一次，返回第一个输出（转成 float） */
    private fun run(session: OrtSession, s: Spec): FloatArray {
        val env = LocalAi.env
        val tensors = s.inputs.map { (name, shape) ->
            val n = shape.fold(1L) { a, b -> a * b }.toInt()
            name to if (name == "image" || name == "mask") {
                val buf = ByteBuffer.allocateDirect(n).order(ByteOrder.nativeOrder())
                if (name == "mask") {
                    val side = shape[3].toInt()
                    for (i in 0 until n) {
                        val x = i % side
                        val y = i / side % side
                        buf.put(if (x in side / 3 until side * 2 / 3 && y in side / 3 until side * 2 / 3) 0 else 255.toByte())
                    }
                } else {
                    for (i in 0 until n) buf.put(((128 + 100 * sin(i * 0.013)).toInt()).toByte())
                }
                buf.rewind()
                OnnxTensor.createTensor(env, buf, shape, OnnxJavaType.UINT8)
            } else {
                OnnxTensor.createTensor(env, FloatBuffer.wrap(AccelMath.signal(n, name == "source")), shape)
            }
        }
        try {
            session.run(tensors.toMap()).use { r ->
                val t = r.get(0) as OnnxTensor
                return if (t.info.type == OnnxJavaType.UINT8) {
                    val b = t.byteBuffer
                    FloatArray(b.remaining()) { (b.get().toInt() and 0xff).toFloat() }
                } else {
                    val f = t.floatBuffer
                    FloatArray(f.remaining()).also { f.get(it) }
                }
            }
        } finally {
            tensors.forEach { it.second.close() }
        }
    }
}

/** 加速选择里的纯计算（可单测） */
object AccelMath {
    /** 确定性的测试输入：图像是 0..1 的平滑波纹；[unitVector] 时归一化成单位向量（换脸模型的 source） */
    fun signal(n: Int, unitVector: Boolean): FloatArray {
        val v = FloatArray(n) { (0.5 + 0.45 * sin(it * 0.0137 + (it / 97) * 0.31)).toFloat() }
        if (!unitVector) return v
        for (i in v.indices) v[i] -= 0.5f
        var s = 0.0
        for (x in v) s += x * x
        val k = (1 / kotlin.math.sqrt(s)).toFloat()
        for (i in v.indices) v[i] *= k
        return v
    }

    /** 和 CPU 的结果一致：平均误差不到值域的 2%，最大误差不到 30%（半精度会有些出入，但不能整块出错） */
    fun agrees(ref: FloatArray, other: FloatArray): Boolean {
        if (ref.size != other.size || ref.isEmpty()) return false
        var lo = Float.MAX_VALUE
        var hi = -Float.MAX_VALUE
        for (x in ref) { if (x < lo) lo = x; if (x > hi) hi = x }
        val range = (hi - lo).coerceAtLeast(1e-6f)
        var sum = 0.0
        var max = 0f
        for (i in ref.indices) {
            val d = abs(ref[i] - other[i])
            if (d.isNaN()) return false
            sum += d
            if (d > max) max = d
        }
        return sum / ref.size / range < 0.02 && max / range < 0.3
    }

    /** 选总耗时最短的；要比 CPU 快 10% 以上才换，避免测量误差带来来回切换 */
    fun pick(score: Map<Accel.Backend, Double>): Accel.Backend {
        val cpu = score[Accel.Backend.CPU] ?: return Accel.Backend.CPU
        val best = score.minByOrNull { it.value } ?: return Accel.Backend.CPU
        return if (best.value < cpu * 0.9) best.key else Accel.Backend.CPU
    }
}
