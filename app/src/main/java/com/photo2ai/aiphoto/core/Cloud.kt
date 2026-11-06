package com.photo2ai.aiphoto.core

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import com.photo2ai.aiphoto.core.CloudProtocol.Failure
import com.photo2ai.aiphoto.core.Http.await
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.math.max

/**
 * 云端 AI 修图：用户自己的 Gemini 或 OpenAI 兼容接口。照片缩到长边 1536 再上传。
 */
object Cloud {
    private const val UPLOAD_SIDE = 1536

    private val client by lazy {
        Http.client.newBuilder()
            .writeTimeout(60, TimeUnit.SECONDS)
            .readTimeout(180, TimeUnit.SECONDS)
            .callTimeout(240, TimeUnit.SECONDS)
            .build()
    }

    /** 按描述修图。返回的图尺寸由模型决定（通常长边 1024 左右） */
    suspend fun edit(cfg: Store.Cloud, src: Bitmap, prompt: String): Bitmap = withContext(Dispatchers.Default) {
        val img = ImageIO.fit(src, UPLOAD_SIDE)
        val bytes = when (cfg.provider) {
            CloudProtocol.OPENAI -> openAi(cfg, img, prompt, null)
            else -> gemini(cfg, img, CloudProtocol.editPrompt(prompt))
        }
        decode(bytes)
    }

    /**
     * AI 消除：[mask] 是任意尺寸的 ALPHA_8 蒙版（和原图同比例）。只把蒙版区域（略外扩、羽化）从结果里取回来贴到原图上，
     * 其它地方保留原图的像素和分辨率。
     */
    suspend fun erase(cfg: Store.Cloud, src: Bitmap, mask: Bitmap): Bitmap = withContext(Dispatchers.Default) {
        val img = ImageIO.fit(src, UPLOAD_SIDE)
        val m = ImageIO.scaled(mask, img.width, img.height)
        val bytes = when (cfg.provider) {
            CloudProtocol.OPENAI -> openAi(cfg, img, CloudProtocol.ERASE_PROMPT_MASK, openAiMask(m))
            else -> gemini(cfg, marked(img, m), CloudProtocol.ERASE_PROMPT_MARKED)
        }
        val result = ImageIO.scaled(decode(bytes), src.width, src.height)
        val full = alphaOf(ImageIO.scaled(mask, src.width, src.height))
        val r = max(2, max(src.width, src.height) / 150)
        val grown = Pixels.blurF(full, src.width, src.height, r, 2)
        for (i in grown.indices) grown[i] = maxOf(full[i], (grown[i] * 3f).coerceAtMost(1f))
        ImageIO.toBitmap(Pixels.composite(ImageIO.pixels(result), ImageIO.pixels(src), grown), src.width, src.height)
    }

    private suspend fun gemini(cfg: Store.Cloud, img: Bitmap, prompt: String): ByteArray {
        val body = CloudProtocol.geminiBody(prompt, ImageIO.encode(img, png = false, quality = 92), "image/jpeg")
        val req = Request.Builder()
            .url(CloudProtocol.geminiUrl(cfg.base, cfg.model))
            .header("x-goog-api-key", cfg.key)
            .header("User-Agent", Http.UA)
            .post(body.toRequestBody("application/json".toMediaType()))
            .build()
        return CloudProtocol.parseGemini(send(req))
    }

    private suspend fun openAi(cfg: Store.Cloud, img: Bitmap, prompt: String, mask: ByteArray?): ByteArray {
        val png = "image/png".toMediaType()
        val form = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("model", cfg.model)
            .addFormDataPart("prompt", prompt)
            .addFormDataPart("image", "image.png", ImageIO.encode(img, png = true).toRequestBody(png))
        if (mask != null) form.addFormDataPart("mask", "mask.png", mask.toRequestBody(png))
        val req = Request.Builder()
            .url(CloudProtocol.openAiUrl(cfg.base))
            .header("Authorization", "Bearer " + cfg.key)
            .header("User-Agent", Http.UA)
            .post(form.build())
            .build()
        val (bytes, url) = CloudProtocol.parseOpenAi(send(req))
        if (bytes != null) return bytes
        return withContext(Dispatchers.IO) {
            client.newCall(Request.Builder().url(url!!).build()).await().use { r ->
                if (!r.isSuccessful) throw Failure("下载结果图失败（HTTP ${r.code}）")
                r.body?.bytes() ?: throw Failure("下载结果图失败")
            }
        }
    }

    private suspend fun send(req: Request): String = withContext(Dispatchers.IO) {
        try {
            client.newCall(req).await().use { r ->
                val text = r.body?.string().orEmpty()
                if (!r.isSuccessful) throw Failure(CloudProtocol.httpError(r.code, text))
                text
            }
        } catch (e: IOException) {
            throw Failure("网络错误：" + (e.message ?: e.javaClass.simpleName))
        } catch (e: IllegalArgumentException) {
            throw Failure("接口地址不对：" + req.url)
        }
    }

    private fun decode(bytes: ByteArray): Bitmap {
        val b = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: throw Failure("返回的图片无法解码")
        return ImageIO.ensureArgb(b)
    }

    /** 在要去掉的地方盖一层半透明红色 */
    private fun marked(img: Bitmap, mask: Bitmap): Bitmap {
        val out = (if (img.hasAlpha()) ImageIO.flattenWhite(img) else img.copy(Bitmap.Config.ARGB_8888, true))
        val tint = Bitmap.createBitmap(img.width, img.height, Bitmap.Config.ARGB_8888)
        Canvas(tint).apply {
            drawBitmap(mask, 0f, 0f, Paint().apply { color = Color.RED })
        }
        Canvas(out).drawBitmap(tint, 0f, 0f, Paint().apply { alpha = 150 })
        return out
    }

    /** OpenAI 的蒙版：和原图同尺寸的 PNG，完全透明处是要改的地方 */
    private fun openAiMask(mask: Bitmap): ByteArray {
        val out = Bitmap.createBitmap(mask.width, mask.height, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawColor(Color.BLACK)
            drawBitmap(mask, 0f, 0f, Paint().apply { xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_OUT) })
        }
        return ImageIO.encode(out, png = true)
    }

    /** ALPHA_8 蒙版 → 0..1 */
    fun alphaOf(mask: Bitmap): FloatArray {
        val px = ImageIO.pixels(mask)
        return FloatArray(px.size) { (px[it] ushr 24) / 255f }
    }
}
