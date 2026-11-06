package com.photo2ai.aiphoto.ui

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import com.photo2ai.aiphoto.BuildConfig
import com.photo2ai.aiphoto.R
import com.photo2ai.aiphoto.core.CloudProtocol
import com.photo2ai.aiphoto.core.Store

class SettingsActivity : AppCompatActivity() {
    private lateinit var key: TextInputEditText
    private lateinit var base: TextInputEditText
    private lateinit var model: TextInputEditText
    private var shown = CloudProtocol.GEMINI

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        findViewById<android.view.View>(R.id.root).applySystemInsets()
        findViewById<android.view.View>(R.id.back).setOnClickListener { finish() }
        key = findViewById(R.id.key)
        base = findViewById(R.id.base)
        model = findViewById(R.id.model)

        val group = findViewById<MaterialButtonToggleGroup>(R.id.provider)
        shown = Store.provider
        group.check(idOf(shown))
        load()
        group.addOnButtonCheckedListener { _, id, checked ->
            if (!checked) return@addOnButtonCheckedListener
            save()
            shown = when (id) {
                R.id.providerOpenai -> CloudProtocol.OPENAI
                R.id.providerGemini -> CloudProtocol.GEMINI
                else -> CloudProtocol.SEEDREAM
            }
            Store.provider = shown
            load()
        }

        findViewById<TextView>(R.id.version).text = "当前版本 " + BuildConfig.VERSION_NAME
        findViewById<android.view.View>(R.id.checkUpdate).setOnClickListener { UpdateUi.manualCheck(this) }
    }

    /** 每个服务的 Key、地址、模型分开存，切换时各自保留 */
    private fun load() {
        key.setText(Store.key(shown))
        base.setText(Store.rawBase(shown))
        model.setText(Store.rawModel(shown))
        findViewById<TextInputLayout>(R.id.baseLayout).placeholderText = Store.defaultBase(shown)
        findViewById<TextInputLayout>(R.id.modelLayout).placeholderText = Store.defaultModel(shown)
        findViewById<TextView>(R.id.providerHelp).text = when (shown) {
            CloudProtocol.OPENAI ->
                "默认模型 ${Store.defaultModel(shown)}，调用 /images/edits 接口；兼容这个接口的第三方服务（中转、自建）把地址改成它的 …/v1 即可。"
            CloudProtocol.GEMINI ->
                "默认模型 ${Store.defaultModel(shown)}（可以换成更新的图片模型），Key 在 Google AI Studio 申请。国内需要代理，或把接口地址改成中转地址。"
            else ->
                "国内直连。Key 在火山方舟控制台「API Key 管理」里创建，开通 Seedream 模型后新用户有免费额度（以控制台为准），之后按张计费。" +
                    "默认模型 ${Store.defaultModel(shown)}，也可以填其它已开通的 Seedream 版本或推理接入点 ID（ep-…）。"
        }
    }

    private fun idOf(provider: String) = when (provider) {
        CloudProtocol.OPENAI -> R.id.providerOpenai
        CloudProtocol.GEMINI -> R.id.providerGemini
        else -> R.id.providerSeedream
    }

    private fun save() {
        Store.setKey(shown, key.text?.toString().orEmpty())
        Store.setBase(shown, base.text?.toString().orEmpty())
        Store.setModel(shown, model.text?.toString().orEmpty())
    }

    override fun onPause() {
        super.onPause()
        save()
    }
}
