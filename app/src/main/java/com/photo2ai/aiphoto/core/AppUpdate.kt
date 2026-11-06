package com.photo2ai.aiphoto.core

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInfo
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.core.content.FileProvider
import com.photo2ai.aiphoto.BuildConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import okhttp3.Call
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

/**
 * 应用内更新（移植自 TV Player）。版本说明是仓库里的 `releases/app-version.json`：
 * `{"versionCode": 2, "versionName": "1.0.1", "url": "https://…/app.apk", "sha256": "…", "size": 123, "notes": "…", "force": false}`.
 * The downloaded file is checked against `sha256`, and Android itself refuses to install an update that is not
 * signed with the same key as the installed app, so a tampered download cannot replace the app.
 */
object AppUpdate {
    class Info(val code: Int, val name: String, val url: String, val sha256: String, val notes: String, val size: Long, val force: Boolean)

    class Check(val info: Info?, val error: String?)

    /**
     * The default source, and the same file through other GitHub mirrors and GitHub itself. Mirrors differ in speed and in
     * how long they keep an old copy, so they are asked at the same time.
     */
    private const val GITHUB_PATH = "photo2ai/ai-photo/main/releases/app-version.json"
    private val MIRRORS = mirrorsOf("https://gh-proxy.com/https://raw.githubusercontent.com/$GITHUB_PATH")

    /**
     * The same GitHub file through each mirror. [url] is a gh-proxy address of the form
     * `https://gh-proxy.com/https://raw.githubusercontent.com/<repo path>`; other addresses are used as they are.
     */
    fun mirrorsOf(url: String): List<String> {
        val prefix = "https://gh-proxy.com/https://raw.githubusercontent.com/"
        if (!url.startsWith(prefix)) return listOf(url)
        val path = url.removePrefix(prefix)
        return listOf(
            prefix + path,
            "https://ghfast.top/https://raw.githubusercontent.com/" + path,
            "https://cdn.jsdelivr.net/gh/" + path.replaceFirst("/main/", "@main/"),
            "https://raw.githubusercontent.com/" + path,
            "https://ghproxy.net/https://raw.githubusercontent.com/" + path,
        )
    }

    /** After the first usable answer, the others get this long: a fresher copy from a slower mirror may still win. */
    private const val GRACE_MS = 3_000L
    private val fetchScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** [info] is null when this version is current, was skipped by the user, or the manifest is missing. */
    suspend fun check(respectIgnored: Boolean): Check {
        val (o, error) = newestManifest(MIRRORS)
        if (o == null) return Check(null, error)
        val code = o.optInt("versionCode", 0)
        if (code <= BuildConfig.VERSION_CODE) return Check(null, null)
        val url = o.optString("url")
        if (!url.startsWith("http://") && !url.startsWith("https://")) return Check(null, "更新信息里没有下载地址")
        if (!o.optString("sha256").matches(Regex("[0-9a-fA-F]{64}"))) return Check(null, "更新信息缺少校验值")
        val info = Info(
            code, o.optString("versionName").ifEmpty { code.toString() }, url, o.optString("sha256").lowercase(),
            o.optString("notes"), o.optLong("size"), o.optBoolean("force"),
        )
        if (respectIgnored && !info.force && Store.ignoredVersion == code) return Check(null, null)
        return Check(info, null)
    }

    /**
     * The manifest with the highest versionCode among the answers. Waits for the first usable answer, then at most
     * [GRACE_MS] more for the rest; the requests still open then are cancelled. A 404 (no manifest there) is not an error.
     */
    private suspend fun newestManifest(urls: List<String>): Pair<JSONObject?, String?> {
        val answers = Channel<Pair<JSONObject?, String?>>(urls.size)
        val jobs = urls.map { url -> fetchScope.launch { answers.send(fetchManifest(url)) } }
        var newest: JSONObject? = null
        var error: String? = null
        var answered = 0
        while (answered < urls.size) {
            val (o, e) = if (newest == null) answers.receive() else (withTimeoutOrNull(GRACE_MS) { answers.receive() } ?: break)
            answered++
            if (o == null) {
                error = error ?: e
                continue
            }
            if (newest == null || o.optInt("versionCode", 0) > newest.optInt("versionCode", 0)) newest = o
        }
        jobs.forEach { it.cancel() }
        return newest to error
    }

    /**
     * The first mirror that answers with a file wins: the others are cancelled, so the download comes from the fastest
     * one. Fails only when no mirror has the file.
     */
    private suspend fun openFirst(urls: List<String>, client: OkHttpClient, headers: Map<String, String> = emptyMap()): Response {
        val calls = urls.map { url ->
            client.newCall(Request.Builder().url(url).header("User-Agent", Http.UA).apply { headers.forEach { (k, v) -> header(k, v) } }.build())
        }
        val answers = Channel<Pair<Call, Response?>>(calls.size)
        val jobs = calls.map { call -> fetchScope.launch { answers.send(call to runCatching { call.execute() }.getOrNull()) } }
        var winner: Response? = null
        var lastCode = 0
        try {
            repeat(calls.size) {
                val (call, r) = answers.receive()
                if (r == null) return@repeat
                if (winner == null && r.isSuccessful) {
                    winner = r
                    calls.forEach { other -> if (other !== call) other.cancel() }
                } else {
                    lastCode = r.code
                    r.close()
                }
            }
        } catch (e: CancellationException) {
            // The download was abandoned: stop every request, including a winner nobody will read.
            calls.forEach { it.cancel() }
            winner?.close()
            throw e
        }
        jobs.forEach { it.cancel() }
        return winner ?: error("下载失败（" + (if (lastCode > 0) "HTTP $lastCode" else "所有线路都连不上") + "）")
    }

    private suspend fun fetchManifest(url: String): Pair<JSONObject?, String?> {
        val r = Http.get(url, timeoutMs = 15_000)
        if (r.status == 404) return null to null
        if (r.status != 200) return null to (r.error ?: "HTTP ${r.status}")
        return try {
            JSONObject(r.text) to null
        } catch (e: JSONException) {
            null to "更新信息格式不对"
        }
    }

    private fun updatesDir(context: Context) = File(context.cacheDir, "updates").apply { mkdirs() }

    /**
     * A finished download of [info] still in the cache, checked against the manifest's hash again (the same version
     * code may have been re-published); null when there is none. Hashes the file: call it off the main thread.
     */
    fun ready(context: Context, info: Info): File? {
        val file = File(updatesDir(context), "app-${info.code}.apk")
        if (!file.isFile || file.length() == 0L || (info.size > 0 && file.length() != info.size)) return null
        if (info.sha256.isNotEmpty() && sha256(file) != info.sha256) {
            file.delete()
            return null
        }
        return file
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { s ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = s.read(buf)
                if (n < 0) break
                digest.update(buf, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    /** Bytes a broken-off download of [info] left behind; the next download goes on from there. */
    fun partialBytes(context: Context, info: Info): Long = File(updatesDir(context), "app-${info.code}.apk.part").length()

    /**
     * Downloads into the cache directory and verifies the hash. Throws with a readable message on failure. A download
     * that broke off (network gone, app killed, cancelled) left `.part` behind: it is continued with a Range request,
     * and the bytes already there are hashed first so the check still covers the whole file. A mirror that ignores
     * the range answers 200 with the whole file, which then simply starts over.
     */
    suspend fun download(context: Context, info: Info, onProgress: (done: Long, total: Long) -> Unit): File =
        withContext(Dispatchers.IO) {
            val dir = updatesDir(context)
            val part = File(dir, "app-${info.code}.apk.part")
            val file = File(dir, "app-${info.code}.apk")
            dir.listFiles()?.forEach { if (it != part) it.delete() } // other versions, or a copy that failed its check
            var have = part.length()
            if (info.size in 1..have) {
                part.delete() // at least as big as the whole file: not a usable remainder
                have = 0
            }
            val digest = MessageDigest.getInstance("SHA-256")
            val buf = ByteArray(64 * 1024)
            if (have > 0) part.inputStream().use { s ->
                while (true) {
                    ensureActive()
                    val n = s.read(buf)
                    if (n < 0) break
                    digest.update(buf, 0, n)
                }
            }
            val client = Http.client.newBuilder().readTimeout(30, TimeUnit.SECONDS).build()
            val headers = if (have > 0) mapOf("Range" to "bytes=$have-") else emptyMap()
            val response = try {
                openFirst(mirrorsOf(info.url), client, headers)
            } catch (e: IllegalStateException) {
                // 416: no mirror can continue from here (the remainder is not of this file): start over without it.
                if (have > 0 && e.message?.contains("416") == true) {
                    part.delete()
                    return@withContext download(context, info, onProgress)
                }
                throw e
            }
            response.use { r ->
                if (!r.isSuccessful) error("下载失败（HTTP ${r.code}）")
                val body = r.body ?: error("下载失败：没有内容")
                val resuming = have > 0 && r.code == 206
                android.util.Log.d("AIPhoto", "update download: HTTP ${r.code} from ${r.request.url.host}, had $have bytes, resuming=$resuming")
                if (!resuming) {
                    digest.reset()
                    have = 0
                }
                val total = body.contentLength().takeIf { it > 0 }?.let { it + have } ?: info.size
                body.byteStream().use { input ->
                    java.io.FileOutputStream(part, resuming).use { out ->
                        var done = have
                        onProgress(done, total)
                        while (true) {
                            ensureActive()
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            digest.update(buf, 0, n)
                            done += n
                            onProgress(done, total)
                        }
                    }
                }
                val hex = digest.digest().joinToString("") { "%02x".format(it) }
                if (info.sha256.isNotEmpty() && hex != info.sha256) {
                    part.delete()
                    error("校验失败：下载的文件和发布信息对不上，已丢弃")
                }
            }
            if (!part.renameTo(file)) error("保存更新文件失败")
            file
        }

    fun canInstall(context: Context) = context.packageManager.canRequestPackageInstalls()

    /** Opens the system installer; the caller must have checked [canInstall]. */
    fun install(activity: Activity, file: File) {
        val uri = FileProvider.getUriForFile(activity, "${activity.packageName}.files", file)
        activity.startActivity(
            Intent(Intent.ACTION_VIEW)
                .setDataAndType(uri, "application/vnd.android.package-archive")
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION),
        )
    }

    /**
     * 安装前再核对一次：必须是本应用的包、签名证书和已安装的一致。更新信息来自第三方镜像，被篡改时可能换成别的应用的包
     * 诱导用户安装（换不了本应用：系统会拒绝签名不同的覆盖安装，但「装一个新应用」系统不拦）。返回 null 表示通过。
     */
    fun verifyPackage(context: Context, file: File): String? {
        val pm = context.packageManager
        @Suppress("DEPRECATION")
        val flags = if (Build.VERSION.SDK_INT >= 28) PackageManager.GET_SIGNING_CERTIFICATES else PackageManager.GET_SIGNATURES
        val archive = pm.getPackageArchiveInfo(file.path, flags) ?: return "安装包无法解析"
        if (archive.packageName != context.packageName) return "安装包不是本应用，已丢弃"
        val mine = certs(pm.getPackageInfo(context.packageName, flags))
        val theirs = certs(archive)
        if (mine.isEmpty() || !theirs.containsAll(mine)) return "安装包签名和当前应用不一致，已丢弃"
        return null
    }

    private fun certs(info: PackageInfo): Set<String> {
        @Suppress("DEPRECATION")
        val sigs = if (Build.VERSION.SDK_INT >= 28) {
            val si = info.signingInfo ?: return emptySet()
            if (si.hasMultipleSigners()) si.apkContentsSigners else si.signingCertificateHistory
        } else {
            info.signatures
        } ?: return emptySet()
        return sigs.map { sig -> MessageDigest.getInstance("SHA-256").digest(sig.toByteArray()).joinToString("") { "%02x".format(it) } }.toSet()
    }

    /** 有的系统没有这个设置页，返回 false 时由调用方告诉用户去哪里开 */
    fun openInstallPermissionSettings(activity: Activity): Boolean = try {
        activity.startActivity(Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${activity.packageName}")))
        true
    } catch (e: android.content.ActivityNotFoundException) {
        false
    }
}
