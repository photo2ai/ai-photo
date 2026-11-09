package com.photo2ai.aiphoto.core

import android.content.Context
import android.os.Build
import kotlinx.coroutines.CancellationException
import java.io.File

/**
 * 本地模型：放在仓库的 models/ 目录，第一次用到时下载到 files/models，按 SHA-256 校验（和应用内更新同样的镜像和续传）。
 * 改了 models/ 里的文件，要同步改这里的校验值和大小。
 */
object LocalModels {
    /**
     * [hfPath] 是模型在 Hugging Face 上的原始位置（仓库/resolve/版本/文件），null 表示 Hugging Face 上没有（我们自己量化的），
     * 只从本仓库下。[inRepo] 为 true 时本仓库 models/ 里也有一份（同一个文件），作为最后的下载来源；换脸的两个大模型只从
     * Hugging Face 下。[replaces] 是这个模型取代的旧文件，新的下好后从手机上删掉。
     */
    class Model(
        val file: String, val sha256: String, val size: Long, val title: String, val hfPath: String?,
        val inRepo: Boolean = true, val replaces: List<String> = emptyList(),
    )

    val MIGAN = Model(
        "migan-pipeline-v2.onnx", "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b", 28_079_181, "AI 消除模型",
        "andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx",
    )
    /**
     * 变清晰：Real-ESRGAN 的 INT8 量化版（体积约 3.5 分之一，CPU 上快 2 倍多，和原模型的输出差 40 dB 以上，看不出来）。
     * 做法见 models/README.md。Hugging Face 上没有，只从本仓库下（1.4 MB，gh-proxy 慢的时候也就一两分钟）。
     */
    val ESRGAN = Model(
        "realesr-general-x4v3-int8.onnx", "da89a87d6254e20c97751e708645bc4d7daf15840be2c54f65ee36cd12f3bc2b", 1_406_805, "变清晰模型", null,
        replaces = listOf("realesr-general-x4v3.onnx"),
    )

    /**
     * 高清换脸：ArcFace 身份特征模型的 INT8 量化版（InsightFace，仅限非商业使用）。原版 174 MB；量化版和原版同一张脸的特征
     * 相似度 0.99。放在我们自己的 Hugging Face 仓库（scripts/quantize_faceswap.py 生成，做法见 models/README.md）
     */
    val ARCFACE = Model(
        "arcface_w600k_r50_int8.onnx", "43deedbae06600d32b569046c29bb9aee94f34d1be79a88f033e653383f96d8d", 43_977_929, "人脸特征模型",
        "Lakes0001/ai-photo-models/resolve/main/arcface_w600k_r50_int8.onnx", inRepo = false, replaces = listOf("arcface_w600k_r50.onnx"),
    )

    /**
     * 高清换脸：HyperSwap 1c（FaceFusion 团队训练，256×256；原版半精度 403 MB）只量化权重的 INT8 版（和原版差 51 dB 以上）。
     * 换出的脸比 inswapper 干净、更像本人，后面仍接 GFPGAN 修复。scripts/convert_hyperswap.py 生成，放在我们的 Hugging Face 仓库；
     * 模型许可没有写明
     */
    val HYPERSWAP = Model(
        "hyperswap_1c_256_w8.onnx", "702ec8e5dd1b45a89abe366c5815262e49789ba770353e29c5043de036e5b949", 202_061_901, "换脸模型",
        "Lakes0001/ai-photo-models/resolve/main/hyperswap_1c_256_w8.onnx", inRepo = false,
        replaces = listOf("inswapper_128_fp16.onnx", "inswapper_128_int8.onnx"),
    )

    /**
     * 高清换脸的人脸修复：GFPGAN 1.4 的半精度版（原版 340 MB，输出和原版差 60 dB 以上）。换脸模型只输出 128×128，
     * 不修复又糊又假（眼睛、牙齿有瑕疵）。Apache 2.0 许可。scripts/convert_gfpgan_fp16.py 生成，放在我们的 Hugging Face 仓库
     */
    val GFPGAN = Model(
        "gfpgan_1.4_fp16.onnx", "a94d1ae4353e9fee4f7dae2639bf9b3311188eab3d40b9ec8cc821e8aa42db3c", 170_261_734, "人脸修复模型",
        "Lakes0001/ai-photo-models/resolve/main/gfpgan_1.4_fp16.onnx", inRepo = false,
    )

    /** 所有本地模型（设置页、本机 AI 加速、启动清理都按这个列表） */
    val ALL get() = listOf(MIGAN, ESRGAN, ARCFACE, HYPERSWAP, GFPGAN)

    private const val BASE = "https://gh-proxy.com/https://raw.githubusercontent.com/photo2ai/ai-photo/main/models/"

    /** ONNX Runtime 只打了 arm64 的库（见 app/build.gradle），32 位手机用不了本地模型 */
    fun supported(): Boolean = Build.SUPPORTED_64_BIT_ABIS.contains("arm64-v8a")

    private fun dir(context: Context) = File(context.filesDir, "models").apply { mkdirs() }

    private fun deleteWithMarkers(d: File, name: String) {
        for (suffix in listOf("", ".ok", ".part")) File(d, name + suffix).delete()
    }

    /**
     * 删掉被新模型取代的旧文件（[Model.replaces]），启动时调用：新版 App 用不到它们，老用户升级后不该等到下载新模型
     * 才腾出空间（换脸的旧模型有 450 MB）
     */
    fun removeReplaced(context: Context) {
        val d = File(context.filesDir, "models")
        if (!d.isDirectory) return
        for (m in ALL) m.replaces.forEach { deleteWithMarkers(d, it) }
    }

    /**
     * 已经下载好就返回文件，否则 null。下载时校验通过才写 `.ok` 标记（内容是校验值），这里认标记，
     * 不每次重算 28MB 的哈希；模型加载失败时调用 [discard] 删掉，下次重新下载。
     */
    fun ready(context: Context, m: Model): File? {
        val f = File(dir(context), m.file)
        val ok = File(dir(context), m.file + ".ok")
        return f.takeIf { it.isFile && it.length() == m.size && ok.isFile && ok.readText() == m.sha256 }
    }

    /**
     * 依次试：hf-mirror（国内直连实测 500KB/s 以上）→ Hugging Face 原地址 → 本仓库的 GitHub 镜像（几个镜像同时问）。
     * GitHub 镜像国内直连只有 10～20KB/s，28MB 要半个多小时，只作兜底。换来源时已下的部分接着续传（同一个文件）。
     */
    suspend fun download(context: Context, m: Model, onProgress: (done: Long, total: Long) -> Unit): File {
        val d = dir(context)
        val part = File(d, m.file + ".part")
        val target = File(d, m.file)
        // hf-mirror 中途超时多是一时的：先再试一次它，不行再换（GitHub 镜像国内只有十几 KB/s，放最后）
        val hf = m.hfPath?.let { listOf("https://hf-mirror.com/$it", "https://hf-mirror.com/$it", "https://huggingface.co/$it") } ?: emptyList()
        val repo = if (m.inRepo) listOf(BASE + m.file) else emptyList()
        // 只在本仓库的模型：镜像一时连不上就再试一轮
        val sources = hf + repo + (if (hf.isEmpty()) repo else emptyList())
        var last: Exception? = null
        for (url in sources) {
            try {
                val f = AppUpdate.downloadVerified(url, m.sha256, m.size, part, target, onProgress)
                File(d, m.file + ".ok").writeText(m.sha256)
                m.replaces.forEach { deleteWithMarkers(d, it) }
                return f
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                android.util.Log.w("AIPhoto", "model download failed from $url: ${e.message}")
                last = e
            }
        }
        throw last ?: IllegalStateException("下载失败")
    }

    /** 上次中断留下的字节数，下次从这里续传 */
    fun partialBytes(context: Context, m: Model): Long = File(dir(context), m.file + ".part").length()

    /** 模型文件坏了（加载失败）：删掉，下次用时重新下载 */
    fun discard(file: File) {
        File(file.path + ".ok").delete()
        file.delete()
    }
}
