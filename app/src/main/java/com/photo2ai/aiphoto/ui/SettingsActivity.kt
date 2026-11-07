package com.photo2ai.aiphoto.ui

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import com.photo2ai.aiphoto.BuildConfig
import com.photo2ai.aiphoto.R
import androidx.lifecycle.lifecycleScope
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.materialswitch.MaterialSwitch
import com.photo2ai.aiphoto.core.Accel
import com.photo2ai.aiphoto.core.CloudProtocol
import com.photo2ai.aiphoto.core.LocalAi
import com.photo2ai.aiphoto.core.LocalModels
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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

        val accel = findViewById<MaterialSwitch>(R.id.accel)
        accel.isChecked = Store.accelEnabled
        accel.setOnCheckedChangeListener { _, on ->
            Store.accelEnabled = on
            LocalAi.releaseAll()
            showAccel()
        }
        findViewById<android.view.View>(R.id.accelTest).setOnClickListener { retest() }
        showAccel()

        findViewById<TextView>(R.id.version).text = "当前版本 " + BuildConfig.VERSION_NAME
        findViewById<android.view.View>(R.id.checkUpdate).setOnClickListener { UpdateUi.manualCheck(this) }
    }

    private val localModels = listOf(LocalModels.MIGAN, LocalModels.ESRGAN, LocalModels.ARCFACE, LocalModels.INSWAPPER)

    private fun showAccel() {
        val lines = localModels.filter { LocalModels.ready(this, it) != null }.map { m ->
            Store.accelSummary(m.sha256).ifEmpty { m.title + "：还没测，第一次使用时自动测" }
        }
        val head = when {
            !LocalModels.supported() -> "这台手机是 32 位系统，用不了本地模型。"
            !Store.accelEnabled -> "已关闭，本地模型都用 CPU 计算。结果不对或者闪退时可以关掉试试。"
            else -> "每个模型第一次使用时，在这台手机上分别测一次 CPU 和 AI 芯片，结果正确又更快的才用。"
        }
        findViewById<TextView>(R.id.accelInfo).text =
            head + if (lines.isEmpty()) "\n还没有下载本地模型。" else "\n" + lines.joinToString("\n")
        findViewById<android.view.View>(R.id.accelTest).isEnabled = Store.accelEnabled && lines.isNotEmpty() && LocalModels.supported()
    }

    /** 已下载的模型全部重新测一遍 */
    private fun retest() {
        val models = localModels.mapNotNull { m -> LocalModels.ready(this, m)?.let { m to it } }
        val text = TextView(this).apply { setPadding(dp(24), dp(16), dp(24), 0) }
        var job: Job? = null
        val dialog = MaterialAlertDialogBuilder(this)
            .setTitle("正在测试")
            .setView(text)
            .setCancelable(false)
            .setNegativeButton("停止") { _, _ -> job?.cancel() }
            .show()
        job = lifecycleScope.launch {
            LocalAi.releaseAll()
            Store.clearAccel()
            for ((k, pair) in models.withIndex()) {
                val (m, f) = pair
                text.text = "${m.title}（${k + 1}/${models.size}）…\n大模型要几十秒"
                withContext(Dispatchers.Default) {
                    try {
                        Accel.benchmark(m, f)
                    } catch (e: Exception) {
                        Store.setAccel(m.sha256, Accel.Backend.CPU.name, m.title + "：CPU（测速出错）")
                    }
                }
            }
            dialog.dismiss()
            showAccel()
        }
        job.invokeOnCompletion { runOnUiThread { if (dialog.isShowing) dialog.dismiss(); showAccel() } }
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
