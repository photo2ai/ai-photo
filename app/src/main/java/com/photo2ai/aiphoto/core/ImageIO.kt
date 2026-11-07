package com.photo2ai.aiphoto.core

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

object ImageIO {
    const val ALBUM = "AI修图"

    /** 编辑用的最大像素数：按可用内存估，编辑时同时存在约 10 份整图大小的数据，最多 1600 万像素 */
    fun maxPixels(): Long = min(16_000_000L, Runtime.getRuntime().maxMemory() / 40)

    /** 读入并按 EXIF 摆正，超过 [maxPixels] 的缩小 */
    fun decode(context: Context, uri: Uri): Bitmap {
        val cr = context.contentResolver
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        // 只读尺寸时 decodeStream 本来就返回 null，打不开要看流本身
        val input = cr.openInputStream(uri) ?: throw IOException("打不开这个文件")
        input.use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw IOException("不是能识别的图片格式")
        val limit = maxPixels()
        var sample = 1
        while ((bounds.outWidth.toLong() / (sample * 2)) * (bounds.outHeight / (sample * 2)) >= limit) sample *= 2
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        var bmp = cr.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) } ?: throw IOException("图片解码失败")
        val pixels = bmp.width.toLong() * bmp.height
        if (pixels > limit) {
            val s = sqrt(limit.toDouble() / pixels)
            bmp = scaled(bmp, (bmp.width * s).roundToInt(), (bmp.height * s).roundToInt())
        }
        val orientation = try {
            cr.openInputStream(uri)?.use { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
        } catch (e: Exception) {
            null
        } ?: ExifInterface.ORIENTATION_NORMAL
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
        }
        if (!m.isIdentity) bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        return ensureArgb(bmp)
    }

    fun ensureArgb(b: Bitmap): Bitmap = if (b.config == Bitmap.Config.ARGB_8888) b else b.copy(Bitmap.Config.ARGB_8888, true)

    fun scaled(b: Bitmap, w: Int, h: Int): Bitmap =
        if (b.width == w && b.height == h) b else Bitmap.createScaledBitmap(b, max(1, w), max(1, h), true)

    /** 长边不超过 [maxSide]，本来就小的原样返回 */
    fun fit(b: Bitmap, maxSide: Int): Bitmap {
        val side = max(b.width, b.height)
        if (side <= maxSide) return b
        val s = maxSide.toFloat() / side
        return scaled(b, (b.width * s).roundToInt(), (b.height * s).roundToInt())
    }

    fun pixels(b: Bitmap): IntArray = IntArray(b.width * b.height).also { b.getPixels(it, 0, b.width, 0, 0, b.width, b.height) }

    fun toBitmap(px: IntArray, w: Int, h: Int): Bitmap =
        Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { it.setPixels(px, 0, w, 0, 0, w, h) }

    /** 透明处铺白（存 JPEG、发给只收 JPEG 的接口时用，否则透明会变黑） */
    fun flattenWhite(b: Bitmap): Bitmap {
        val out = Bitmap.createBitmap(b.width, b.height, Bitmap.Config.ARGB_8888)
        Canvas(out).apply { drawColor(Color.WHITE); drawBitmap(b, 0f, 0f, null) }
        return out
    }

    fun encode(b: Bitmap, png: Boolean, quality: Int = 95): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        if (png) b.compress(Bitmap.CompressFormat.PNG, 100, out) else (if (b.hasAlpha()) flattenWhite(b) else b).compress(Bitmap.CompressFormat.JPEG, quality, out)
        return out.toByteArray()
    }

    /** 存进相册的「AI修图」文件夹。有透明像素存 PNG，否则 JPEG。返回显示给用户的位置 */
    fun saveToGallery(context: Context, b: Bitmap): String {
        val png = b.hasAlpha() && Pixels.hasTransparency(pixels(b))
        val name = "AIPhoto_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + if (png) ".png" else ".jpg"
        val mime = if (png) "image/png" else "image/jpeg"
        val bytes = encode(b, png)
        if (Build.VERSION.SDK_INT >= 29) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, mime)
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/" + ALBUM)
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val cr = context.contentResolver
            val uri = cr.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: throw IOException("相册拒绝写入")
            try {
                cr.openOutputStream(uri)?.use { it.write(bytes) } ?: throw IOException("相册拒绝写入")
                cr.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
            } catch (e: Exception) {
                cr.delete(uri, null, null)
                throw e
            }
        } else {
            // Android 8～9：直接写公共目录（调用方已申请存储权限），再通知媒体库
            @Suppress("DEPRECATION")
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), ALBUM).apply { mkdirs() }
            val f = File(dir, name)
            f.writeBytes(bytes)
            @Suppress("DEPRECATION")
            context.sendBroadcast(Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, Uri.fromFile(f)))
        }
        return "相册 / $ALBUM / $name"
    }

    class EditFiles(val input: File, val inputUri: Uri, val output: File, val outputUri: Uri, val mime: String)

    /**
     * 交给系统相册编辑的文件，在缓存目录 share/edit 下：一个是要编辑的照片（有透明像素存 PNG，否则 JPEG），
     * 一个是空的「修好后存到这里」。两个必须分开：有的修图应用一打开就以写方式打开输出文件（会清空它），
     * 和输入是同一个文件的话它就读不到照片了。
     */
    fun editFiles(context: Context, b: Bitmap): EditFiles {
        val png = b.hasAlpha() && Pixels.hasTransparency(pixels(b))
        val ext = if (png) ".png" else ".jpg"
        val dir = File(context.cacheDir, "share/edit").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }
        val stamp = System.currentTimeMillis()
        val input = File(dir, "AIPhoto_$stamp$ext").apply { writeBytes(encode(b, png)) }
        val output = File(dir, "AIPhoto_${stamp}_edited$ext").apply { createNewFile() }
        val auth = context.packageName + ".files"
        return EditFiles(
            input, FileProvider.getUriForFile(context, auth, input),
            output, FileProvider.getUriForFile(context, auth, output),
            if (png) "image/png" else "image/jpeg",
        )
    }

    /** 写到缓存目录，返回给分享用的 content:// 地址 */
    fun shareUri(context: Context, b: Bitmap): Uri {
        val png = b.hasAlpha() && Pixels.hasTransparency(pixels(b))
        val dir = File(context.cacheDir, "share").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }
        val f = File(dir, "AIPhoto_" + System.currentTimeMillis() + if (png) ".png" else ".jpg")
        f.writeBytes(encode(b, png))
        return FileProvider.getUriForFile(context, context.packageName + ".files", f)
    }
}
