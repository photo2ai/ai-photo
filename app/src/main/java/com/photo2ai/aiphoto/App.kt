package com.photo2ai.aiphoto

import android.app.Application
import com.photo2ai.aiphoto.core.Accel
import com.photo2ai.aiphoto.core.LocalAi
import com.photo2ai.aiphoto.core.LocalModels
import com.photo2ai.aiphoto.core.Store

class App : Application() {
    override fun onCreate() {
        super.onCreate()
        Store.init(this)
        LocalAi.backendFor = { Accel.chosen(it) }
        Thread { LocalModels.removeReplaced(this) }.start()
    }
}
