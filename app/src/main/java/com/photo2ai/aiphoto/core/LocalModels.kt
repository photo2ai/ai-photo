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
     * [hfPath] 是模型在 Hugging Face 上的原始位置（仓库/resolve/版本/文件）。[inRepo] 为 true 时本仓库 models/ 里也有一份
     * （同一个文件），作为最后的下载来源；换脸的两个大模型只从 Hugging Face 下。
     */
    class Model(val file: String, val sha256: String, val size: Long, val title: String, val hfPath: String, val inRepo: Boolean = true)

    val MIGAN = Model(
        "migan-pipeline-v2.onnx", "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b", 28_079_181, "AI 消除模型",
        "andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx",
    )
    val ESRGAN = Model(
        "realesr-general-x4v3.onnx", "7b10e80021ec3e0ff35017ae5022791879d93a799b2e201e1010a37510202266", 4_866_420, "变清晰模型",
        "vlad-grig/realesr-general-x4v3-onnx/resolve/main/onnx/model.onnx",
    )

    /** 高清换脸：ArcFace 身份特征模型（InsightFace，仅限非商业使用） */
    val ARCFACE = Model(
        "arcface_w600k_r50.onnx", "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1", 174_388_474, "人脸特征模型",
        "facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx", inRepo = false,
    )

    /** 高清换脸：inswapper 半精度（InsightFace，仅限非商业使用） */
    val INSWAPPER = Model(
        "inswapper_128_fp16.onnx", "c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420", 277_680_829, "换脸模型",
        "facefusion/models-3.0.0/resolve/main/inswapper_128_fp16.onnx", inRepo = false,
    )

    private const val BASE = "https://gh-proxy.com/https://raw.githubusercontent.com/photo2ai/ai-photo/main/models/"

    /** ONNX Runtime 只打了 arm64 的库（见 app/build.gradle），32 位手机用不了本地模型 */
    fun supported(): Boolean = Build.SUPPORTED_64_BIT_ABIS.contains("arm64-v8a")

    private fun dir(context: Context) = File(context.filesDir, "models").apply { mkdirs() }

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
        val sources = listOf("https://hf-mirror.com/" + m.hfPath, "https://huggingface.co/" + m.hfPath) + (if (m.inRepo) listOf(BASE + m.file) else emptyList())
        var last: Exception? = null
        for (url in sources) {
            try {
                val f = AppUpdate.downloadVerified(url, m.sha256, m.size, part, target, onProgress)
                File(d, m.file + ".ok").writeText(m.sha256)
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

    /** 模型文件坏了（加载失败）：删掉，下次用时重新下载 */
    fun discard(file: File) {
        File(file.path + ".ok").delete()
        file.delete()
    }
}
