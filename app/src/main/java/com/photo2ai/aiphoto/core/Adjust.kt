package com.photo2ai.aiphoto.core

import kotlin.math.max
import kotlin.math.pow

/**
 * 一组调整参数，范围都是 -1..1（褪色、锐化是 0..1）。滤镜也是一组预设参数，强度滑杆就是把参数整体缩放。
 */
data class Adjust(
    val brightness: Float = 0f,
    val contrast: Float = 0f,
    val highlights: Float = 0f,
    val shadows: Float = 0f,
    val saturation: Float = 0f,
    val warmth: Float = 0f,
    val tint: Float = 0f,
    val fade: Float = 0f,
    val sharpen: Float = 0f,
    val vignette: Float = 0f,
) {
    val isIdentity get() = this == Adjust()

    fun scaled(t: Float) = Adjust(
        brightness * t, contrast * t, highlights * t, shadows * t, saturation * t,
        warmth * t, tint * t, fade * t, sharpen * t, vignette * t,
    )

    /** 亮度、对比度、高光、阴影、褪色合成一张查找表，三个通道共用 */
    fun toneLut(): IntArray = IntArray(256) { i ->
        var x = i / 255f
        if (brightness != 0f) x = x.pow(2f.pow(-brightness))
        // 阴影在 1/3 处、高光在 2/3 处影响最大；系数保证曲线单调
        if (shadows != 0f) x += shadows * 0.14f * 6.75f * x * (1 - x) * (1 - x)
        if (highlights != 0f) x += highlights * 0.14f * 6.75f * x * x * (1 - x)
        if (contrast != 0f) x = 0.5f + (x - 0.5f) * (1f + contrast * 0.6f)
        if (fade > 0f) {
            val k = 0.22f * fade
            x = k + x.coerceIn(0f, 1f) * (1 - k)
        }
        Pixels.clamp255(x * 255f)
    }

    /** 作用到整张图。锐化半径按图的尺寸换算，预览图和原图的观感接近 */
    fun apply(px: IntArray, w: Int, h: Int): IntArray {
        if (isIdentity) return px.copyOf()
        val lut = toneLut()
        val rk = 1f + 0.15f * warmth
        val bk = 1f - 0.15f * warmth
        val gk = 1f - 0.12f * tint
        val sat = 1f + saturation
        var out = IntArray(px.size)
        for (i in px.indices) {
            val c = px[i]
            var r = lut[c shr 16 and 0xff] * rk
            var g = lut[c shr 8 and 0xff] * gk
            var b = lut[c and 0xff] * bk
            if (sat != 1f) {
                val l = 0.299f * r + 0.587f * g + 0.114f * b
                r = l + (r - l) * sat
                g = l + (g - l) * sat
                b = l + (b - l) * sat
            }
            out[i] = Pixels.rgb(c, Pixels.clamp255(r), Pixels.clamp255(g), Pixels.clamp255(b))
        }
        if (sharpen > 0f) out = Pixels.sharpen(out, w, h, sharpen * 1.5f, max(1, max(w, h) / 2000))
        if (vignette != 0f) out = Pixels.vignette(out, w, h, vignette)
        return out
    }
}

object Filters {
    class Filter(val name: String, val adjust: Adjust)

    val ALL = listOf(
        Filter("原图", Adjust()),
        Filter("鲜明", Adjust(contrast = .25f, saturation = .35f, sharpen = .2f)),
        Filter("暖阳", Adjust(brightness = .08f, warmth = .45f, saturation = .1f)),
        Filter("清冷", Adjust(warmth = -.4f, tint = -.05f, contrast = .1f, highlights = .1f)),
        Filter("胶片", Adjust(contrast = -.1f, saturation = -.15f, warmth = .15f, fade = .35f, vignette = -.2f)),
        Filter("日系", Adjust(brightness = .25f, contrast = -.2f, saturation = -.2f, warmth = -.12f, fade = .15f, shadows = .3f)),
        Filter("复古", Adjust(warmth = .35f, saturation = -.35f, fade = .25f, vignette = -.4f, tint = .05f)),
        Filter("浓郁", Adjust(contrast = .2f, saturation = .55f, shadows = .2f, highlights = -.2f)),
        Filter("黑白", Adjust(saturation = -1f, contrast = .25f)),
        Filter("褪色黑白", Adjust(saturation = -1f, contrast = -.1f, fade = .3f)),
    )
}
