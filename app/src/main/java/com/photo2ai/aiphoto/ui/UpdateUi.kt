package com.photo2ai.aiphoto.ui

import android.text.format.Formatter
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.photo2ai.aiphoto.core.AppUpdate
import com.photo2ai.aiphoto.core.Store
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** 检查更新、下载、安装的界面部分 */
object UpdateUi {
    private const val AUTO_INTERVAL = 12 * 3600_000L

    /** 打开 App 时，每 12 小时最多查一次；没新版、出错都不打扰 */
    fun autoCheck(a: AppCompatActivity) {
        val now = System.currentTimeMillis()
        if (now - Store.lastUpdateCheck < AUTO_INTERVAL) return
        Store.lastUpdateCheck = now
        a.lifecycleScope.launch {
            val c = AppUpdate.check(respectIgnored = true)
            c.info?.let { offer(a, it) }
        }
    }

    fun manualCheck(a: AppCompatActivity) {
        a.toast("正在检查更新…")
        a.lifecycleScope.launch {
            val c = AppUpdate.check(respectIgnored = false)
            Store.lastUpdateCheck = System.currentTimeMillis()
            when {
                c.info != null -> offer(a, c.info)
                c.error != null -> a.toast("检查失败：" + c.error, long = true)
                else -> a.toast("已经是最新版本")
            }
        }
    }

    private fun offer(a: AppCompatActivity, info: AppUpdate.Info) {
        if (a.isFinishing) return
        val size = if (info.size > 0) "（" + Formatter.formatShortFileSize(a, info.size) + "）" else ""
        val b = MaterialAlertDialogBuilder(a)
            .setTitle("发现新版本 ${info.name}")
            .setMessage(info.notes.ifBlank { "有新版本可以更新。" } + "\n\n下载安装包$size")
            .setPositiveButton("更新") { _, _ -> download(a, info) }
            .setCancelable(!info.force)
        if (!info.force) {
            b.setNegativeButton("以后", null)
            b.setNeutralButton("跳过这个版本") { _, _ -> Store.ignoredVersion = info.code }
        }
        b.show()
    }

    private fun download(a: AppCompatActivity, info: AppUpdate.Info) {
        val bar = LinearProgressIndicator(a).apply { isIndeterminate = true; max = 1000 }
        val text = TextView(a).apply { setPadding(0, a.dp(10), 0, 0) }
        val box = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(a.dp(24), a.dp(16), a.dp(24), 0)
            addView(bar)
            addView(text)
        }
        var lastPermille = -1
        var job: Job? = null
        val dialog = MaterialAlertDialogBuilder(a)
            .setTitle("正在下载 ${info.name}")
            .setView(box)
            .setCancelable(false)
            .setNegativeButton("取消") { _, _ -> job?.cancel() }
            .show()
        job = a.lifecycleScope.launch {
            try {
                val file = AppUpdate.ready(a, info) ?: AppUpdate.download(a, info) { done, total ->
                    val p = if (total > 0) (done * 1000 / total).toInt() else -1
                    if (p == lastPermille) return@download
                    lastPermille = p
                    a.runOnUiThread {
                        if (p >= 0) {
                            bar.isIndeterminate = false
                            bar.progress = p
                        }
                        text.text = Formatter.formatShortFileSize(a, done) + if (total > 0) " / " + Formatter.formatShortFileSize(a, total) else ""
                    }
                }
                withContext(Dispatchers.IO) { AppUpdate.verifyPackage(a, file) }?.let {
                    file.delete()
                    error(it)
                }
                dialog.dismiss()
                install(a, file)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                dialog.dismiss()
                a.toast(e.message ?: "下载失败", long = true)
            }
        }
    }

    private fun install(a: AppCompatActivity, file: java.io.File) {
        if (AppUpdate.canInstall(a)) {
            AppUpdate.install(a, file)
            return
        }
        MaterialAlertDialogBuilder(a)
            .setTitle("需要允许安装")
            .setMessage("请在接下来的系统页面里允许「AI 修图」安装应用，返回后再点一次「检查更新」→「更新」（已下载的文件会直接用）。")
            .setPositiveButton("去允许") { _, _ ->
                if (!AppUpdate.openInstallPermissionSettings(a)) a.toast("请在系统设置 → 应用 → 特殊权限 → 安装未知应用里允许", long = true)
            }
            .setNegativeButton("取消", null)
            .show()
    }
}
