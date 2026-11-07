package com.photo2ai.aiphoto.core

import android.content.Context
import android.content.SharedPreferences

/** 本机设置。API Key 只存在本机（应用关闭了系统备份） */
object Store {
    private lateinit var p: SharedPreferences

    fun init(context: Context) {
        p = context.getSharedPreferences("aiphoto", Context.MODE_PRIVATE)
        // 1.0.2 起默认服务从 Gemini 改成火山引擎。以前没切换过服务（没存 provider）、但填了 Gemini Key 的，继续用 Gemini
        if (!p.contains("provider") && key(CloudProtocol.GEMINI).isNotEmpty()) provider = CloudProtocol.GEMINI
    }

    class Cloud(val provider: String, val key: String, val base: String, val model: String)

    fun defaultBase(provider: String) = when (provider) {
        CloudProtocol.OPENAI -> "https://api.openai.com/v1"
        CloudProtocol.GEMINI -> "https://generativelanguage.googleapis.com"
        else -> "https://ark.cn-beijing.volces.com/api/v3"
    }

    fun defaultModel(provider: String) = when (provider) {
        CloudProtocol.OPENAI -> "gpt-image-1"
        CloudProtocol.GEMINI -> "gemini-2.5-flash-image"
        else -> "doubao-seedream-4-0-250828"
    }

    fun providerName(provider: String) = when (provider) {
        CloudProtocol.OPENAI -> "OpenAI 兼容接口"
        CloudProtocol.GEMINI -> "Google Gemini"
        else -> "火山引擎 豆包 Seedream"
    }

    /** 默认火山引擎：国内直连，新用户有免费额度 */
    var provider: String
        get() = p.getString("provider", CloudProtocol.SEEDREAM) ?: CloudProtocol.SEEDREAM
        set(v) = p.edit().putString("provider", v).apply()

    fun key(provider: String) = p.getString("key_$provider", "").orEmpty()
    fun setKey(provider: String, v: String) = p.edit().putString("key_$provider", v.trim()).apply()

    /** 空字符串表示用默认值 */
    fun base(provider: String) = p.getString("base_$provider", "").orEmpty().ifBlank { defaultBase(provider) }
    fun setBase(provider: String, v: String) = p.edit().putString("base_$provider", v.trim()).apply()
    fun rawBase(provider: String) = p.getString("base_$provider", "").orEmpty()

    fun model(provider: String) = p.getString("model_$provider", "").orEmpty().ifBlank { defaultModel(provider) }
    fun setModel(provider: String, v: String) = p.edit().putString("model_$provider", v.trim()).apply()
    fun rawModel(provider: String) = p.getString("model_$provider", "").orEmpty()

    /** 当前选中的云端服务；没填 Key 时为 null */
    fun cloud(): Cloud? {
        val pr = provider
        val k = key(pr)
        if (k.isEmpty()) return null
        return Cloud(pr, k, base(pr), model(pr))
    }

    var ignoredVersion: Int
        get() = p.getInt("ignored_version", 0)
        set(v) = p.edit().putInt("ignored_version", v).apply()

    var lastUpdateCheck: Long
        get() = p.getLong("last_update_check", 0)
        set(v) = p.edit().putLong("last_update_check", v).apply()

    /** 本地模型用不用 AI 芯片加速（默认用；关了一律 CPU） */
    var accelEnabled: Boolean
        get() = p.getBoolean("accel_enabled", true)
        set(v) = p.edit().putBoolean("accel_enabled", v).apply()

    /** 某个模型（按校验值区分）在这台手机上测出来该用的方式；没测过为 null */
    fun accelChoice(sha: String) = p.getString("accel_" + sha.take(16), null)
    fun accelSummary(sha: String) = p.getString("accel_sum_" + sha.take(16), "").orEmpty()
    fun setAccel(sha: String, backend: String, summary: String) =
        p.edit().putString("accel_" + sha.take(16), backend).putString("accel_sum_" + sha.take(16), summary).apply()

    fun clearAccel() {
        val e = p.edit()
        for (k in p.all.keys) if (k.startsWith("accel_") && k != "accel_enabled") e.remove(k)
        e.apply()
    }

    /** 用户确认过「换脸只用于本人或已获同意的照片」 */
    var faceConsent: Boolean
        get() = p.getBoolean("face_consent", false)
        set(v) = p.edit().putBoolean("face_consent", v).apply()

    /** 修图描述的最近几条，最新的在前 */
    fun recentPrompts(): List<String> = p.getString("recent_prompts", "").orEmpty().split('\n').filter { it.isNotBlank() }

    fun addRecentPrompt(s: String) {
        val list = (listOf(s.trim()) + recentPrompts().filter { it != s.trim() }).take(6)
        p.edit().putString("recent_prompts", list.joinToString("\n")).apply()
    }
}
