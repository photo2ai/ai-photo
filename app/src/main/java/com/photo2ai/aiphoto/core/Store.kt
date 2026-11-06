package com.photo2ai.aiphoto.core

import android.content.Context
import android.content.SharedPreferences

/** 本机设置。API Key 只存在本机（应用关闭了系统备份） */
object Store {
    private lateinit var p: SharedPreferences

    fun init(context: Context) {
        p = context.getSharedPreferences("aiphoto", Context.MODE_PRIVATE)
    }

    class Cloud(val provider: String, val key: String, val base: String, val model: String)

    fun defaultBase(provider: String) =
        if (provider == CloudProtocol.OPENAI) "https://api.openai.com/v1" else "https://generativelanguage.googleapis.com"

    fun defaultModel(provider: String) = if (provider == CloudProtocol.OPENAI) "gpt-image-1" else "gemini-2.5-flash-image"

    fun providerName(provider: String) = if (provider == CloudProtocol.OPENAI) "OpenAI 兼容接口" else "Google Gemini"

    var provider: String
        get() = p.getString("provider", CloudProtocol.GEMINI) ?: CloudProtocol.GEMINI
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

    /** 修图描述的最近几条，最新的在前 */
    fun recentPrompts(): List<String> = p.getString("recent_prompts", "").orEmpty().split('\n').filter { it.isNotBlank() }

    fun addRecentPrompt(s: String) {
        val list = (listOf(s.trim()) + recentPrompts().filter { it != s.trim() }).take(6)
        p.edit().putString("recent_prompts", list.joinToString("\n")).apply()
    }
}
