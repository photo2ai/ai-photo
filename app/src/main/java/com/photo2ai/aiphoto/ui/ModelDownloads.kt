package com.photo2ai.aiphoto.ui

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.widget.Toast
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.photo2ai.aiphoto.R
import com.photo2ai.aiphoto.core.LocalModels
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * 本地模型的下载，不绑在页面上（和应用更新同样的做法）：在自己的协程作用域里跑，前台服务 [ModelDownloadService] 保活，
 * 页面上的进度弹窗可以收起（「后台下载」），收起后由通知显示进度；下完发通知。中断（断网、杀进程、取消）留下的部分
 * 下次接着下（续传在 AppUpdate.downloadVerified）。同时只下一批。
 */
object ModelDownloads {
    const val NOTIFY_ID = 4
    private const val DONE_ID = 5
    private const val CHANNEL = "models"

    /** 当前进度。[error] 不为 null 表示失败，[finished] 表示这一批结束了（成功或失败、取消） */
    class State(val title: String, val step: Int, val steps: Int, val done: Long, val total: Long, val finished: Boolean = false, val error: String? = null) {
        val percent get() = if (total > 0) (done * 100 / total).toInt() else -1
        val label get() = title + (if (steps > 1) "（$step/$steps）" else "")
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val main = Handler(Looper.getMainLooper())
    private var job: Job? = null
    private var lastNotified = 0L

    var models: List<LocalModels.Model> = emptyList()
        private set
    var state: State? = null
        private set

    /** 页面上的进度弹窗；收起（后台下载）时设为 null，之后由通知显示 */
    var listener: ((State) -> Unit)? = null

    val running get() = job?.isActive == true

    fun start(context: Context, list: List<LocalModels.Model>) {
        if (running || list.isEmpty()) return
        val app = context.applicationContext
        NotificationManagerCompat.from(app).cancel(DONE_ID)
        models = list
        update(app, State(list[0].title, 1, list.size, LocalModels.partialBytes(app, list[0]), list[0].size))
        runCatching { ContextCompat.startForegroundService(app, Intent(app, ModelDownloadService::class.java)) }
        job = scope.launch {
            try {
                list.forEachIndexed { k, m ->
                    LocalModels.download(app, m) { done, total ->
                        main.post { update(app, State(m.title, k + 1, list.size, done, total)) }
                    }
                }
                stop(app)
                val s = State("模型", list.size, list.size, 1, 1, finished = true)
                update(app, s)
                if (listener == null) {
                    notifyDone(app, "模型已下载好", "回到 AI 修图再点一次就能用了")
                    Toast.makeText(app, "模型已下载好，可以使用了", Toast.LENGTH_SHORT).show()
                }
            } catch (e: CancellationException) {
                stop(app)
                update(app, State("模型", 0, list.size, 0, 0, finished = true, error = "已取消"))
            } catch (e: Exception) {
                stop(app)
                val kept = list.sumOf { LocalModels.partialBytes(app, it) }
                val msg = (e.message ?: "下载失败") + if (kept > 0) "\n已下载 %.1f MB，下次会接着下".format(Locale.ROOT, kept / 1048576.0) else ""
                update(app, State("模型", 0, list.size, 0, 0, finished = true, error = msg))
                if (listener == null) notifyDone(app, "模型下载中断", "再用到这个功能时会接着下")
            }
        }
    }

    fun cancel() {
        job?.cancel()
    }

    private fun stop(app: Context) {
        app.stopService(Intent(app, ModelDownloadService::class.java))
        NotificationManagerCompat.from(app).cancel(NOTIFY_ID)
    }

    private fun update(app: Context, s: State) {
        state = s
        val l = listener
        if (l != null) {
            l(s)
            return
        }
        if (s.finished) return
        // 没有弹窗：更新通知，大约一秒一次
        val now = System.currentTimeMillis()
        if (now - lastNotified < 1_000) return
        lastNotified = now
        post(app, NOTIFY_ID, progressNotification(app))
    }

    /** 弹窗收起后马上刷一次通知 */
    fun detached(app: Context) {
        listener = null
        if (running) post(app.applicationContext, NOTIFY_ID, progressNotification(app.applicationContext))
    }

    fun progressNotification(app: Context): Notification {
        val s = state
        val pct = s?.percent ?: -1
        return build(app, "正在下载" + (s?.label ?: "模型"), if (pct >= 0) "$pct%" else "", ongoing = true, progress = pct)
    }

    private fun notifyDone(app: Context, title: String, text: String) = post(app, DONE_ID, build(app, title, text, ongoing = false, progress = -1))

    private fun post(app: Context, id: Int, n: Notification) {
        val nm = NotificationManagerCompat.from(app)
        if (!nm.areNotificationsEnabled()) return
        try {
            nm.notify(id, n)
        } catch (e: SecurityException) {
            // 没给通知权限
        }
    }

    private fun build(app: Context, title: String, text: String, ongoing: Boolean, progress: Int): Notification {
        if (Build.VERSION.SDK_INT >= 26) {
            app.getSystemService(NotificationManager::class.java)
                .createNotificationChannel(NotificationChannel(CHANNEL, "模型下载", NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(
            app, 1, Intent(app, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
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

/** 下载模型时在前台运行，App 切到后台也不断网；通知就是下载进度 */
class ModelDownloadService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val n = ModelDownloads.progressNotification(this)
        if (Build.VERSION.SDK_INT >= 29) startForeground(ModelDownloads.NOTIFY_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        else startForeground(ModelDownloads.NOTIFY_ID, n)
        return START_NOT_STICKY
    }

    /** 安卓 15 对 dataSync 前台服务限时：到点就停，已下的部分留着下次续传 */
    override fun onTimeout(startId: Int) {
        ModelDownloads.cancel()
        stopSelf()
    }
}
