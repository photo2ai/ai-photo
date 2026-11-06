package com.photo2ai.aiphoto.core

import com.photo2ai.aiphoto.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

object Http {
    const val UA = "AIPhoto/" + BuildConfig.VERSION_NAME + " (Android)"

    val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    class Resp(val status: Int, val text: String, val url: String, val error: String? = null)

    suspend fun get(url: String, timeoutMs: Long = 15_000): Resp = withContext(Dispatchers.IO) {
        try {
            val c = client.newBuilder().callTimeout(timeoutMs, TimeUnit.MILLISECONDS).build()
            c.newCall(Request.Builder().url(url).header("User-Agent", UA).build()).await().use {
                Resp(it.code, it.body?.string().orEmpty(), url)
            }
        } catch (e: IOException) {
            Resp(0, "", url, e.message ?: e.javaClass.simpleName)
        } catch (e: IllegalArgumentException) {
            Resp(0, "", url, "地址不对：$url")
        }
    }

    /** 协程取消时同时取消请求（execute() 是阻塞的，取消不了） */
    suspend fun Call.await(): Response = suspendCancellableCoroutine { cont ->
        cont.invokeOnCancellation { cancel() }
        enqueue(object : Callback {
            override fun onResponse(call: Call, response: Response) {
                cont.resume(response) { _, value, _ -> value.close() }
            }

            override fun onFailure(call: Call, e: IOException) {
                if (!cont.isCancelled) cont.resumeWithException(e)
            }
        })
    }
}
