package com.photo2ai.aiphoto.core

import android.content.Context
import android.os.Build
import java.io.File

/**
 * 本地模型：放在仓库的 models/ 目录，第一次用到时下载到 files/models，按 SHA-256 校验（和应用内更新同样的镜像和续传）。
 * 改了 models/ 里的文件，要同步改这里的校验值和大小。
 */
object LocalModels {
    class Model(val file: String, val sha256: String, val size: Long, val title: String)

    val MIGAN = Model(
        "migan-pipeline-v2.onnx", "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b", 28_079_181, "AI 消除模型",
    )
    val ESRGAN = Model(
        "realesr-general-x4v3.onnx", "7b10e80021ec3e0ff35017ae5022791879d93a799b2e201e1010a37510202266", 4_866_420, "变清晰模型",
    )

    private const val BASE = "https://gh-proxy.com/https://raw.githubusercontent.com/photo2ai/ai-photo/main/models/"

    /** ONNX Runtime 只打了 arm64 的库（见 app/build.gradle），32 位手机用不了本地模型 */
    fun supported(): Boolean = Build.SUPPORTED_64_BIT_ABIS.contains("arm64-v8a")

    private fun dir(context: Context) = File(context.filesDir, "models").apply { mkdirs() }

    /** 已经下载好（下载时校验过）就返回文件，否则 null */
    fun ready(context: Context, m: Model): File? = File(dir(context), m.file).takeIf { it.isFile && it.length() == m.size }

    suspend fun download(context: Context, m: Model, onProgress: (done: Long, total: Long) -> Unit): File {
        val d = dir(context)
        return AppUpdate.downloadVerified(BASE + m.file, m.sha256, m.size, File(d, m.file + ".part"), File(d, m.file), onProgress)
    }
}
