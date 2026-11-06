package com.photo2ai.aiphoto

import android.app.Application
import com.photo2ai.aiphoto.core.Store

class App : Application() {
    override fun onCreate() {
        super.onCreate()
        Store.init(this)
    }
}
