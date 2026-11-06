package com.photo2ai.aiphoto.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AppUpdateTest {
    @Test
    fun mirrorsCoverJsdelivrAndRaw() {
        val m = AppUpdate.mirrorsOf("https://gh-proxy.com/https://raw.githubusercontent.com/photo2ai/ai-photo/main/releases/app-version.json")
        assertTrue(m.contains("https://cdn.jsdelivr.net/gh/photo2ai/ai-photo@main/releases/app-version.json"))
        assertTrue(m.contains("https://raw.githubusercontent.com/photo2ai/ai-photo/main/releases/app-version.json"))
        assertEquals(listOf("https://x/y"), AppUpdate.mirrorsOf("https://x/y"))
    }
}
