package com.photo2ai.aiphoto.core

import android.graphics.Bitmap
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import java.nio.ByteOrder
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.max

/**
 * 人像分割（ML Kit 自拍分割，模型打在包里，本机运行，不联网）。
 * 背景处理的几种效果也在这里：都先在预览尺寸上算，确认时再按原图尺寸算一遍。
 */
object Segment {
    class Mask(val data: FloatArray, val w: Int, val h: Int) {
        fun resized(dw: Int, dh: Int) = if (dw == w && dh == h) data else Pixels.resizeF(data, w, h, dw, dh)
    }

    /** 分割在长边 1024 的图上做，蒙版用导向滤波按这张图的边缘细化 */
    suspend fun person(src: Bitmap): Mask {
        val small = ImageIO.fit(src, 1024)
        val segmenter = Segmentation.getClient(
            SelfieSegmenterOptions.Builder().setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE).build(),
        )
        try {
            val r = segmenter.process(InputImage.fromBitmap(small, 0)).await()
            val buf = r.buffer
            buf.rewind()
            val conf = FloatArray(r.width * r.height)
            buf.order(ByteOrder.nativeOrder()).asFloatBuffer().get(conf)
            val sized = Pixels.resizeF(conf, r.width, r.height, small.width, small.height)
            val refined = Cutout.refine(ImageIO.pixels(small), sized, small.width, small.height)
            return Mask(refined, small.width, small.height)
        } finally {
            segmenter.close()
        }
    }

    enum class Effect(val label: String) {
        BLUR("背景虚化"), MONO("背景黑白"), CLEAR("透明背景"),
        WHITE("白底"), BLUE("证件蓝"), RED("证件红"), GRAY("浅灰"), BLACK("黑底")
    }

    fun colorOf(e: Effect) = when (e) {
        Effect.WHITE -> 0xFFFFFF
        Effect.BLUE -> 0x438EDB
        Effect.RED -> 0xD9001B
        Effect.GRAY -> 0xE6E6E6
        Effect.BLACK -> 0x000000
        else -> 0
    }

    /** 按效果出图。[blur] 0..1 是虚化强度 */
    fun render(src: Bitmap, mask: Mask, effect: Effect, blur: Float): Bitmap {
        val w = src.width
        val h = src.height
        val px = ImageIO.pixels(src)
        val m = mask.resized(w, h)
        val out = when (effect) {
            Effect.BLUR -> Pixels.composite(px, ImageIO.pixels(blurredBackground(src, mask, blur)), m)
            Effect.MONO -> Pixels.composite(px, Pixels.gray(px), m)
            Effect.CLEAR -> Cutout.transparent(px, m)
            else -> Cutout.overColor(px, m, colorOf(effect))
        }
        return ImageIO.toBitmap(out, w, h)
    }

    /**
     * 虚化背景：先在小图上把人像区域用周围背景填掉（不然人像的颜色会晕进背景，出现一圈光晕），再模糊、放大回原尺寸。
     */
    private fun blurredBackground(src: Bitmap, mask: Mask, strength: Float): Bitmap {
        val small = ImageIO.fit(src, 720)
        val sw = small.width
        val sh = small.height
        val spx = ImageIO.pixels(small)
        val sm = mask.resized(sw, sh)
        val filled = Inpaint.fill(spx, sw, sh, Cutout.holeOf(sm, sw, sh))
        val radius = max(1, (max(sw, sh) / 90f * (0.3f + 1.7f * strength)).toInt())
        val blurred = Pixels.boxBlur(filled, sw, sh, radius, 3)
        return ImageIO.scaled(ImageIO.toBitmap(blurred, sw, sh), src.width, src.height)
    }

    private suspend fun <T> Task<T>.await(): T = suspendCancellableCoroutine { cont ->
        addOnSuccessListener { cont.resume(it) }
        addOnFailureListener { cont.resumeWithException(it) }
        addOnCanceledListener { cont.cancel() }
    }
}
