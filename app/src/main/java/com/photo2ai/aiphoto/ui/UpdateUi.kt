package com.photo2ai.aiphoto.ui

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.photo2ai.aiphoto.R
import com.photo2ai.aiphoto.core.AppUpdate
import com.photo2ai.aiphoto.core.Store
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.lang.ref.WeakReference
import java.util.Locale

/**
 * 检查更新、下载、安装的界面部分（移植自 TV Player）。下载不绑在弹窗上：它在自己的协程作用域里跑，弹窗可以收起
 * （「后台下载」），由前台服务 [UpdateService] 保活、通知栏显示进度；下完了 App 在前台就直接弹安装，不在前台就发通知，
 * 下次打开 App 时再提示。中断（断网、杀进程、取消）会保留已下的部分，下次接着下（续传在 [AppUpdate.download]）。
 */
object UpdateUi {
    private const val AUTO_INTERVAL = 12 * 3600_000L
    private const val CHANNEL = "updates"
    const val NOTIFY_ID = 2 // 进度通知（前台服务用）
    private const val DONE_ID = 3 // 「已下载好」「下载中断」：和进度通知分开，服务停止时不会被一起清掉

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val main = Handler(Looper.getMainLooper())

    /** 下完了、在等用户允许「安装未知应用」的安装包 */
    private var pending: File? = null

    /** 下完时 App 不在前台：回到前台再提示安装 */
    private var askOnResume = false

    /** 最近一个和我们打交道的页面：下载结束时弹窗往这里弹 */
    private var attached = WeakReference<AppCompatActivity>(null)

    private class Task(val info: AppUpdate.Info) {
        var job: Job? = null
        var done = 0L
        var total = 0L
        var dialog: AlertDialog? = null
        var bar: LinearProgressIndicator? = null
        var label: TextView? = null
        var lastNotified = 0L
    }

    /** 正在进行的下载（弹窗和通知共用） */
    private var task: Task? = null

    /** 打开 App 时，每 12 小时最多查一次；没新版、出错都不打扰 */
    fun autoCheck(a: AppCompatActivity) {
        attached = WeakReference(a)
        val now = System.currentTimeMillis()
        if (now - Store.lastUpdateCheck < AUTO_INTERVAL) return
        Store.lastUpdateCheck = now
        a.lifecycleScope.launch {
            val c = AppUpdate.check(respectIgnored = true)
            c.info?.let { prompt(a, it) }
        }
    }

    fun manualCheck(a: AppCompatActivity) {
        attached = WeakReference(a)
        task?.let { return showProgress(a, it) } // 已经在下：把进度弹窗再拿出来
        a.toast("正在检查更新…")
        a.lifecycleScope.launch {
            val c = AppUpdate.check(respectIgnored = false)
            Store.lastUpdateCheck = System.currentTimeMillis()
            when {
                c.info != null -> prompt(a, c.info)
                c.error != null -> a.toast("检查失败：" + c.error, long = true)
                else -> a.toast("已经是最新版本")
            }
        }
    }

    /** 有新版：已经下好（之前在后台下完的）就直接问装不装，否则问下不下 */
    private fun prompt(activity: AppCompatActivity, info: AppUpdate.Info) {
        if (activity.isFinishing) return
        task?.takeIf { it.info.code == info.code }?.let { return showProgress(activity, it) }
        activity.lifecycleScope.launch {
            val ready = withContext(Dispatchers.IO) { AppUpdate.ready(activity, info)?.takeIf { AppUpdate.verifyPackage(activity, it) == null } }
            if (ready != null) askInstall(activity, info, ready) else offer(activity, info)
        }
    }

    private fun offer(activity: AppCompatActivity, info: AppUpdate.Info) {
        val partial = AppUpdate.partialBytes(activity, info)
        val size = if (info.size > 0) "大小 %.1f MB".format(Locale.ROOT, info.size / 1048576.0) else ""
        val got = if (partial > 0) "已下载 %.1f MB，会接着下".format(Locale.ROOT, partial / 1048576.0) else ""
        val notes = info.notes.ifBlank { "修复问题，改进体验" }
        val extra = listOf(size, got).filter { it.isNotEmpty() }.joinToString("\n")
        MaterialAlertDialogBuilder(activity)
            .setTitle("发现新版本 ${info.name}")
            .setMessage(if (extra.isEmpty()) notes else "$notes\n\n$extra")
            .setCancelable(!info.force)
            .setPositiveButton(if (partial > 0) "继续下载" else "立即更新") { _, _ -> download(activity, info) }
            .apply {
                if (!info.force) {
                    setNegativeButton("稍后", null)
                    setNeutralButton("忽略此版本") { _, _ -> Store.ignoredVersion = info.code }
                }
            }
            .show()
    }

    private fun askInstall(activity: AppCompatActivity, info: AppUpdate.Info, file: File) {
        MaterialAlertDialogBuilder(activity)
            .setTitle("新版本 ${info.name} 已下载好")
            .setMessage(info.notes.ifBlank { "修复问题，改进体验" })
            .setCancelable(!info.force)
            .setPositiveButton("安装") { _, _ -> installOrAsk(activity, file) }
            .apply { if (!info.force) setNegativeButton("稍后", null) }
            .show()
    }

    private fun download(activity: AppCompatActivity, info: AppUpdate.Info) {
        // 同时只下一个：同一版本就把进度弹窗拿出来，别的版本就取消旧的
        task?.let { running ->
            if (running.info.code == info.code) return showProgress(activity, running)
            running.job?.cancel()
        }
        val app = activity.applicationContext
        if (Build.VERSION.SDK_INT >= 33 && !NotificationManagerCompat.from(app).areNotificationsEnabled()) {
            ActivityCompat.requestPermissions(activity, arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 0)
        }
        NotificationManagerCompat.from(app).cancel(DONE_ID)
        val t = Task(info)
        task = t
        // 安卓 14 起，切到后台变成「缓存」状态的 App 几秒后就断网；前台服务让下载继续，它的通知就是进度通知
        runCatching { ContextCompat.startForegroundService(app, Intent(app, UpdateService::class.java)) }
        t.job = scope.launch {
            try {
                val file = AppUpdate.download(app, info) { done, total ->
                    t.done = done
                    t.total = total
                    main.post { refresh(app, t) }
                }
                // 安装前核对包名和签名证书（版本信息来自第三方镜像）
                withContext(Dispatchers.IO) { AppUpdate.verifyPackage(app, file) }?.let {
                    file.delete()
                    error(it)
                }
                finish(app, t)
                t.dialog?.dismiss()
                pending = file
                val act = attached.get()
                if (act != null && act.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) {
                    installOrAsk(act, file)
                } else {
                    askOnResume = true
                    notify(app, "新版本 ${info.name} 已下载好", "点这里安装", ongoing = false, id = DONE_ID)
                }
            } catch (e: CancellationException) {
                finish(app, t)
                t.dialog?.dismiss()
            } catch (e: Exception) {
                android.util.Log.d("AIPhoto", "update download failed", e)
                finish(app, t)
                failed(app, t, e.message ?: "下载失败")
            }
        }
        showProgress(activity, t)
    }

    private fun finish(app: Context, t: Task) {
        if (task === t) {
            task = null
            app.stopService(Intent(app, UpdateService::class.java))
            NotificationManagerCompat.from(app).cancel(NOTIFY_ID)
        }
    }

    /** [t] 的进度弹窗；「后台下载」收起它，之后由通知显示进度 */
    private fun showProgress(activity: AppCompatActivity, t: Task) {
        t.dialog?.let { old ->
            t.dialog = null // 旧弹窗的关闭回调不能动下面新建的弹窗
            old.dismiss()
        }
        val bar = LinearProgressIndicator(activity).apply { isIndeterminate = true; max = 100 }
        val label = TextView(activity).apply { setPadding(0, activity.dp(10), 0, 0) }
        val box = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(activity.dp(24), activity.dp(16), activity.dp(24), 0)
            addView(bar)
            addView(label)
        }
        t.bar = bar
        t.label = label
        t.dialog = MaterialAlertDialogBuilder(activity)
            .setTitle("正在下载 ${t.info.name}")
            .setView(box)
            .setCancelable(false)
            .setPositiveButton("后台下载", null)
            .setNegativeButton("取消") { _, _ -> t.job?.cancel() }
            .setOnDismissListener { d ->
                if (t.dialog !== d) return@setOnDismissListener
                t.dialog = null
                t.bar = null
                t.label = null
                // 还在下：从现在起由通知显示进度
                if (task === t) refresh(activity.applicationContext, t, force = true)
            }
            .show()
        refresh(activity.applicationContext, t)
    }

    private fun refresh(app: Context, t: Task, force: Boolean = false) {
        val bar = t.bar
        if (bar != null) {
            if (t.total > 0) {
                bar.isIndeterminate = false
                bar.setProgressCompat((t.done * 100 / t.total).toInt(), true)
            }
            t.label?.text = "%.1f / %.1f MB".format(Locale.ROOT, t.done / 1048576.0, maxOf(t.total, t.done) / 1048576.0)
            return
        }
        // 没有弹窗：更新通知，大约一秒一次，免得刷屏
        val now = System.currentTimeMillis()
        if (!force && now - t.lastNotified < 1_000) return
        t.lastNotified = now
        val pct = if (t.total > 0) (t.done * 100 / t.total).toInt() else -1
        notify(app, "正在下载 ${t.info.name}", if (pct >= 0) "$pct%" else "", ongoing = true, progress = pct)
    }

    private fun failed(app: Context, t: Task, message: String) {
        t.dialog?.dismiss()
        val kept = AppUpdate.partialBytes(app, t.info)
        val detail = if (kept > 0) "$message\n已下载 %.1f MB，下次会接着下".format(Locale.ROOT, kept / 1048576.0) else message
        val act = attached.get()
        if (act != null && act.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) {
            MaterialAlertDialogBuilder(act)
                .setTitle("下载中断")
                .setMessage(detail)
                .setPositiveButton("重试") { _, _ -> download(act, t.info) }
                .setNegativeButton("稍后", null)
                .show()
        } else {
            notify(app, "更新下载中断", "打开 AI 修图可以接着下", ongoing = false, id = DONE_ID)
        }
    }

    private fun installOrAsk(activity: AppCompatActivity, file: File) {
        if (!file.isFile) { // 已经被另一个版本的下载清掉
            pending = null
            return
        }
        if (AppUpdate.canInstall(activity)) {
            pending = null
            AppUpdate.install(activity, file)
            return
        }
        pending = file
        MaterialAlertDialogBuilder(activity)
            .setTitle("需要允许安装")
            .setMessage("下载已完成。需要先允许「${activity.getString(R.string.app_name)}」安装应用（只需要允许这一次），返回后会自动继续。")
            .setNegativeButton("取消") { _, _ -> pending = null }
            .setPositiveButton("去允许") { _, _ ->
                if (!AppUpdate.openInstallPermissionSettings(activity)) {
                    MaterialAlertDialogBuilder(activity)
                        .setTitle("请手动允许")
                        .setMessage("请到 系统设置 → 应用 → 特殊应用权限 → 安装未知应用，打开「${activity.getString(R.string.app_name)}」，然后重新点「检查更新」。")
                        .setPositiveButton("知道了", null)
                        .show()
                }
            }
            .show()
    }

    /**
     * 首页、设置页的 onResume 调用：刚允许了安装权限就接着装；在后台下完的就提示安装。
     */
    fun resume(activity: AppCompatActivity) {
        attached = WeakReference(activity)
        val file = pending ?: return
        if (!file.isFile) {
            pending = null
            return
        }
        if (AppUpdate.canInstall(activity)) {
            pending = null
            askOnResume = false
            NotificationManagerCompat.from(activity).cancel(DONE_ID)
            AppUpdate.install(activity, file)
        } else if (askOnResume) {
            askOnResume = false
            NotificationManagerCompat.from(activity).cancel(DONE_ID)
            installOrAsk(activity, file)
        }
    }

    /** 服务时间到了（安卓 15 限制前台服务时长）：停下，已下的部分留着下次续传 */
    fun cancelDownload() {
        task?.job?.cancel()
    }

    /** 当前的进度通知，前台服务启动时用 */
    fun progressNotification(app: Context): Notification {
        val t = task
        val pct = if (t != null && t.total > 0) (t.done * 100 / t.total).toInt() else -1
        return build(app, "正在下载 ${t?.info?.name ?: "更新"}", if (pct >= 0) "$pct%" else "", ongoing = true, progress = pct)
    }

    private fun notify(app: Context, title: String, text: String, ongoing: Boolean, progress: Int = -1, id: Int = NOTIFY_ID) {
        val nm = NotificationManagerCompat.from(app)
        if (!nm.areNotificationsEnabled()) return // 没给通知权限：弹窗和下次打开时的提示照样有
        try {
            nm.notify(id, build(app, title, text, ongoing, progress))
        } catch (e: SecurityException) {
            // 安卓 13+ 没给 POST_NOTIFICATIONS
        }
    }

    private fun build(app: Context, title: String, text: String, ongoing: Boolean, progress: Int): Notification {
        if (Build.VERSION.SDK_INT >= 26) {
            app.getSystemService(NotificationManager::class.java)
                .createNotificationChannel(NotificationChannel(CHANNEL, "应用更新", NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(
            app, 0, Intent(app, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(app, CHANNEL)
            .setSmallIcon(R.drawable.ic_notify)
            .setContentTitle(title)
            .setContentText(text)
            .setOngoing(ongoing)
            .setOnlyAlertOnce(true)
            .setContentIntent(open)
            .setAutoCancel(!ongoing)
            .apply { if (progress >= 0) setProgress(100, progress, false) else if (ongoing) setProgress(0, 0, true) }
            .build()
    }
}
