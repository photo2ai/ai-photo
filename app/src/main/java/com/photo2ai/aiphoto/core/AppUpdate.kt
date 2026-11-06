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
 * `{"versionCode": 2, "versionName": "1.0.1", "url": "https://…/app.apk", "sha256": "…", "size": 123, "notes": "…", "force": false}`。
 * 下载的文件按 `sha256` 校验；安装前再核对包名和签名证书（见 [verifyPackage]），系统本身也拒绝签名不同的覆盖安装。
 */
object AppUpdate {
    class Info(val code: Int, val name: String, val url: String, val sha256: String, val notes: String, val size: Long, val force: Boolean)

    class Check(val info: Info?, val error: String?)

    /** 版本说明的地址：gh-proxy 及其它镜像和 GitHub 原地址。各镜像快慢不同、缓存旧文件的时间也不同，所以同时问 */
    private const val GITHUB_PATH = "photo2ai/ai-photo/main/releases/app-version.json"
    private val MIRRORS = mirrorsOf("https://gh-proxy.com/https://raw.githubusercontent.com/$GITHUB_PATH")

    /**
     * 同一个 GitHub 文件在各个镜像上的地址。[url] 是 `https://gh-proxy.com/https://raw.githubusercontent.com/<仓库路径>`
     * 形式的 gh-proxy 地址；别的地址原样使用。
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

    /** 拿到第一个可用回答后，再等其它镜像这么久：慢一点的镜像可能有更新的版本 */
    private const val GRACE_MS = 3_000L
    private val fetchScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** 已是最新、用户跳过了这个版本、或版本说明不存在时 [info] 为 null */
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
     * 各镜像回答里 versionCode 最高的版本说明。等第一个可用回答，再最多等 [GRACE_MS]，之后还没回的请求取消。
     * 404（那里没有版本说明）不算出错。
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
     * 第一个回应文件的镜像胜出，其它请求取消，所以从最快的镜像下载。所有镜像都没有这个文件才失败；
     * 只要有一个镜像回 416（续传位置不对），错误信息里就带上 416，调用方据此删掉残留重新下载。
     */
    private suspend fun openFirst(urls: List<String>, client: OkHttpClient, headers: Map<String, String> = emptyMap()): Response {
        val calls = urls.map { url ->
            client.newCall(Request.Builder().url(url).header("User-Agent", Http.UA).apply { headers.forEach { (k, v) -> header(k, v) } }.build())
        }
        val answers = Channel<Pair<Call, Response?>>(calls.size)
        val jobs = calls.map { call -> fetchScope.launch { answers.send(call to runCatching { call.execute() }.getOrNull()) } }
        var winner: Response? = null
        var lastCode = 0
        var saw416 = false
        try {
            repeat(calls.size) {
                val (call, r) = answers.receive()
                if (r == null) return@repeat
                if (winner == null && r.isSuccessful) {
                    winner = r
                    calls.forEach { other -> if (other !== call) other.cancel() }
                } else {
                    lastCode = r.code
                    if (r.code == 416) saw416 = true
                    r.close()
                }
            }
        } catch (e: CancellationException) {
            // 下载被放弃：停掉所有请求，已经到了但没人读的响应（包括胜出的那个）都要关掉，否则连接泄漏
            calls.forEach { it.cancel() }
            winner?.close()
            while (true) {
                val (_, r) = answers.tryReceive().getOrNull() ?: break
                r?.close()
            }
            throw e
        }
        jobs.forEach { it.cancel() }
        if (saw416 && winner == null) lastCode = 416
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
     * 缓存里已经下好的 [info] 安装包，按版本说明的校验值再核对一次（同一个版本号可能重新发布过）；没有则 null。
     * 要算整个文件的哈希，不要在主线程调用。
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

    /** [info] 中断的下载已经留下的字节数，下次从这里续传 */
    fun partialBytes(context: Context, info: Info): Long = File(updatesDir(context), "app-${info.code}.apk.part").length()

    /**
     * 下载安装包到缓存目录并校验，失败时抛出可读的错误信息。中断（断网、杀进程、取消）留下的 `.part` 用 Range 续传，
     * 已有的部分先算进哈希，校验仍覆盖整个文件；不支持 Range 的镜像回 200 整个文件，就从头来。
     */
    suspend fun download(context: Context, info: Info, onProgress: (done: Long, total: Long) -> Unit): File {
        val dir = updatesDir(context)
        val part = File(dir, "app-${info.code}.apk.part")
        val file = File(dir, "app-${info.code}.apk")
        withContext(Dispatchers.IO) { dir.listFiles()?.forEach { if (it != part) it.delete() } } // other versions, or a copy that failed its check
        return downloadVerified(info.url, info.sha256, info.size, part, file, onProgress)
    }

    /**
     * 从 [url]（gh-proxy 地址时同时问各个镜像，用最先回应的）下载到 [file]，按 [sha256] 校验整个文件。中断留下的 [part]
     * 下次用 Range 续传。本地模型下载也用它（见 LocalModels）。
     */
    suspend fun downloadVerified(url: String, sha256: String, size: Long, part: File, file: File, onProgress: (done: Long, total: Long) -> Unit): File =
        withContext(Dispatchers.IO) {
            var have = part.length()
            if (size in 1..have) {
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
                openFirst(mirrorsOf(url), client, headers)
            } catch (e: IllegalStateException) {
                // 416：没有镜像能从这里续传（残留的部分不属于这个文件），删掉重新下载
                if (have > 0 && e.message?.contains("416") == true) {
                    part.delete()
                    return@withContext downloadVerified(url, sha256, size, part, file, onProgress)
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
                val total = body.contentLength().takeIf { it > 0 }?.let { it + have } ?: size
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
                if (sha256.isNotEmpty() && hex != sha256) {
                    part.delete()
                    error("校验失败：下载的文件和发布信息对不上，已丢弃")
                }
            }
            file.delete()
            if (!part.renameTo(file)) error("保存下载的文件失败")
            file
        }

    fun canInstall(context: Context) = context.packageManager.canRequestPackageInstalls()

    /** 打开系统安装器；调用前要确认 [canInstall] */
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
