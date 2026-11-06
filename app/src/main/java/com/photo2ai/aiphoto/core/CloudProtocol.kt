package com.photo2ai.aiphoto.core

import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.util.Base64

/** 云端修图接口的请求拼装和响应解析（不碰网络和 Bitmap，能单测） */
object CloudProtocol {
    class Failure(message: String) : Exception(message)

    const val GEMINI = "gemini"
    const val OPENAI = "openai"

    /** 火山引擎方舟的豆包 Seedream（国内直连，新用户有免费额度） */
    const val SEEDREAM = "seedream"

    /** 用户的描述外面包一层：要求只改说到的地方，保持构图 */
    fun editPrompt(user: String) =
        "Edit this photo as follows: ${user.trim()}\n" +
            "Keep the same framing, aspect ratio and everything not mentioned unchanged. Return only the edited image."

    /** Gemini：要去掉的区域在图上涂了半透明红色 */
    const val ERASE_PROMPT_MARKED =
        "The translucent red area marks object(s) to remove from this photo. Remove them completely and fill the area " +
            "realistically so it blends with the surrounding background (continue textures, lines and lighting). " +
            "The result must not contain any red marking. Keep everything else exactly the same, same framing and aspect ratio. " +
            "Return only the edited image."

    /** OpenAI：带蒙版，透明处是要改的区域 */
    const val ERASE_PROMPT_MASK =
        "Remove the object(s) in the masked area and fill it realistically so it blends with the surrounding background. " +
            "Keep everything else unchanged."

    /** 换脸：图 1 是要改的照片，图 2 是提供脸的参考照 */
    const val FACE_SWAP_PROMPT =
        "You are given two images. Image 1 is the photo to edit. Image 2 is a reference photo of a person's face. " +
            "Replace the face of the main person in image 1 with the face of the person in image 2, so that the result clearly " +
            "looks like the person in image 2 (same facial features and face shape). Keep image 1's hairstyle, head angle, " +
            "facial expression, body, clothing, lighting, background and framing unchanged, and match skin tone, lighting and " +
            "grain so the new face blends in seamlessly. Return only the edited image."

    /**
     * 换装。[hasRef] 时图 2 是衣服参考照，[desc] 是补充说明（可空）；没有参考照时 [desc] 就是要换成的衣服。
     */
    fun outfitPrompt(desc: String, hasRef: Boolean): String {
        val keep = "Keep the person's face, hair, pose, body shape, background, lighting and framing unchanged, and make the " +
            "clothes fit the body naturally with realistic folds and shadows. Return only the edited image."
        return if (hasRef) {
            "You are given two images. Image 1 is the photo to edit. Image 2 shows an outfit. Dress the main person in image 1 " +
                "in the outfit from image 2" + (if (desc.isNotBlank()) " (${desc.trim()})" else "") + ". " + keep
        } else {
            "Change the clothing of the main person in this photo to: ${desc.trim()}. " + keep
        }
    }

    /** 地址已经带了版本（…/v1beta、…/v1）就不再加 */
    fun geminiUrl(base: String, model: String): String {
        val b = base.trim().trimEnd('/')
        val versioned = Regex("/v\\d+(alpha|beta)?\\d*$").containsMatchIn(b)
        return b + (if (versioned) "" else "/v1beta") + "/models/" + model.trim() + ":generateContent"
    }

    fun geminiBody(prompt: String, image: ByteArray, mime: String): String = geminiBody(prompt, listOf(image to mime))

    /** 多张图按顺序放在前面（提示词里的 image 1、image 2 就是这个顺序），文字放最后 */
    fun geminiBody(prompt: String, images: List<Pair<ByteArray, String>>): String {
        val parts = JSONArray()
        for ((bytes, mime) in images) {
            parts.put(JSONObject().put("inline_data", JSONObject().put("mime_type", mime).put("data", Base64.getEncoder().encodeToString(bytes))))
        }
        parts.put(JSONObject().put("text", prompt))
        return JSONObject()
            .put("contents", JSONArray().put(JSONObject().put("role", "user").put("parts", parts)))
            .put("generationConfig", JSONObject().put("responseModalities", JSONArray().put("TEXT").put("IMAGE")))
            .toString()
    }

    /** 返回最后一张非「思考过程」的图片字节；没有图时抛出带原因的 [Failure] */
    fun parseGemini(json: String): ByteArray {
        val o = parse(json)
        errorMessage(o)?.let { throw Failure(it) }
        val cands = o.optJSONArray("candidates")
        if (cands == null || cands.length() == 0) {
            val block = o.optJSONObject("promptFeedback")?.optString("blockReason").orEmpty()
            throw Failure(if (block.isNotEmpty()) "请求被服务拒绝（$block）" else "服务没有返回结果")
        }
        val c0 = cands.getJSONObject(0)
        val parts = c0.optJSONObject("content")?.optJSONArray("parts") ?: JSONArray()
        var image: String? = null
        val text = StringBuilder()
        for (i in 0 until parts.length()) {
            val p = parts.optJSONObject(i) ?: continue
            if (p.optBoolean("thought")) continue
            val data = (p.optJSONObject("inlineData") ?: p.optJSONObject("inline_data"))?.optString("data").orEmpty()
            if (data.isNotEmpty()) image = data
            val t = p.optString("text")
            if (t.isNotBlank()) text.append(t.trim()).append(' ')
        }
        if (image != null) return Base64.getMimeDecoder().decode(image)
        val reason = c0.optString("finishReason")
        throw Failure(
            when {
                reason.contains("SAFETY") || reason == "PROHIBITED_CONTENT" -> "请求被服务的安全策略拒绝（$reason）"
                text.isNotBlank() -> "模型没有返回图片：" + text.toString().trim().take(300)
                reason.isNotEmpty() && reason != "STOP" -> "模型没有返回图片（$reason）"
                else -> "模型没有返回图片"
            },
        )
    }

    fun seedreamUrl(base: String) = base.trim().trimEnd('/') + "/images/generations"

    /**
     * Seedream 的输出尺寸：和输入同比例、总像素约 2048×2048，边长取 8 的倍数。
     * 接口要求总像素在 1280×720 到 4096×4096 之间、宽高比在 1/16 到 16 之间。
     */
    fun seedreamSize(w: Int, h: Int): String {
        val aspect = (w.toDouble() / h).coerceIn(1.0 / 16, 16.0)
        val area = 2048.0 * 2048.0
        var ow = Math.sqrt(area * aspect)
        var oh = area / ow
        val s = minOf(1.0, 4096.0 / ow, 4096.0 / oh)
        ow *= s
        oh *= s
        val rw = maxOf(8, (ow / 8).toInt() * 8)
        val rh = maxOf(8, (oh / 8).toInt() * 8)
        return "${rw}x$rh"
    }

    /** 图片用 data URI 内联；一张时 image 是字符串，多张时是数组（按顺序就是提示词里的 image 1、image 2） */
    fun seedreamBody(model: String, prompt: String, jpegs: List<ByteArray>, size: String): String {
        val uris = jpegs.map { "data:image/jpeg;base64," + Base64.getEncoder().encodeToString(it) }
        return JSONObject()
            .put("model", model.trim())
            .put("prompt", prompt)
            .put("image", if (uris.size == 1) uris[0] else JSONArray(uris))
            .put("size", size)
            .put("sequential_image_generation", "disabled")
            .put("response_format", "b64_json")
            .put("watermark", false)
            .toString()
    }

    fun openAiUrl(base: String) = base.trim().trimEnd('/') + "/images/edits"

    /** OpenAI images/edits 和 Seedream（返回格式相同）：返回 (图片字节, null) 或 (null, 图片地址) */
    fun parseOpenAi(json: String): Pair<ByteArray?, String?> {
        val o = parse(json)
        errorMessage(o)?.let { throw Failure(it) }
        val d = o.optJSONArray("data")?.optJSONObject(0) ?: throw Failure("服务没有返回图片")
        val b64 = d.optString("b64_json")
        if (b64.isNotEmpty()) return Base64.getMimeDecoder().decode(b64) to null
        val url = d.optString("url")
        if (url.startsWith("http")) return null to url
        throw Failure("服务没有返回图片")
    }

    /** HTTP 出错时给用户看的话：尽量用服务自己返回的说明 */
    fun httpError(code: Int, body: String): String {
        val msg = runCatching { errorMessage(JSONObject(body)) }.getOrNull()
        val head = when (code) {
            400 -> "请求有误"
            401, 403 -> "API Key 无效或没有权限"
            404 -> "接口地址或模型名不对"
            413 -> "图片太大"
            429 -> "请求太频繁或额度用完了"
            in 500..599 -> "服务暂时出错"
            else -> "请求失败"
        }
        return "$head（HTTP $code）" + (if (!msg.isNullOrBlank()) "：" + msg.take(300) else "")
    }

    private fun parse(json: String): JSONObject = try {
        JSONObject(json)
    } catch (e: JSONException) {
        throw Failure("服务返回的不是 JSON：" + json.take(120))
    }

    private fun errorMessage(o: JSONObject): String? {
        val e = o.opt("error") ?: return null
        return when (e) {
            is JSONObject -> e.optString("message").ifEmpty { e.optString("status") }.ifEmpty { "服务返回错误" }
            is String -> e.ifEmpty { null }
            else -> null
        }
    }
}
