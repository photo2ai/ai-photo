package com.photo2ai.aiphoto.ui

import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder

/**
 * 下载更新时在前台运行，免得进程被当成「缓存」（安卓 14 起，App 离开屏幕几秒后会被断网）。
 * 它的通知就是下载进度；[UpdateUi] 开始下载时启动它，下载结束时停掉。
 */
class UpdateService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = UpdateUi.progressNotification(this)
        if (Build.VERSION.SDK_INT >= 29) startForeground(UpdateUi.NOTIFY_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        else startForeground(UpdateUi.NOTIFY_ID, notification)
        return START_NOT_STICKY
    }

    /** 安卓 15 对 dataSync 前台服务限时：到点就停下，已下的部分留着下次续传 */
    override fun onTimeout(startId: Int) {
        UpdateUi.cancelDownload()
        stopSelf()
    }
}
