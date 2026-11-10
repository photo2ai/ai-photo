package com.photo2ai.aiphoto.core

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt
import kotlin.random.Random

class GlassesTest {
    private val g = GlassesDirection.VECTOR

    private fun dot(a: FloatArray, b: FloatArray) = a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()
    private fun norm(a: FloatArray) = sqrt(dot(a, a))
    private fun unit(a: FloatArray) = norm(a).let { n -> FloatArray(a.size) { a[it] / n } }

    /** 一个随机身份特征，在眼镜方向上的投影正好是 [proj] */
    private fun embWithProjection(proj: Float, seed: Int): FloatArray {
        val r = Random(seed)
        val v = FloatArray(512) { r.nextFloat() - 0.5f }
        val p = dot(v, g)
        val orth = unit(FloatArray(512) { v[it] - p * g[it] })
        val k = sqrt(1 - proj * proj)
        return FloatArray(512) { orth[it] * k + proj * g[it] }
    }

    @Test
    fun directionIsUnitVector() {
        assertEquals(512, g.size)
        assertEquals(1f, norm(g), 1e-3f)
    }

    @Test
    fun removePushesGlassesBelowNoGlassesMean() {
        val e = embWithProjection(0.10f, 1) // 戴眼镜的人：电脑上实测 +0.008～+0.120
        val out = NeuralSwap.adjustGlasses(e, NeuralSwap.Glasses.REMOVE)
        assertEquals(1f, norm(out), 1e-4f)
        assertTrue("投影 ${dot(out, g)}", dot(out, g) < GlassesDirection.NO_GLASSES_MEAN)
        assertTrue("和原特征相似度 ${dot(out, e)}", dot(out, e) > 0.9f)
    }

    @Test
    fun removeLeavesFacesWithoutGlassesAlone() {
        val e = embWithProjection(GlassesDirection.NO_GLASSES_MEAN - 0.05f, 2)
        assertArrayEquals(e, NeuralSwap.adjustGlasses(e, NeuralSwap.Glasses.REMOVE), 1e-5f)
    }

    @Test
    fun keepStrengthensGlasses() {
        val e = embWithProjection(0.05f, 3)
        val out = NeuralSwap.adjustGlasses(e, NeuralSwap.Glasses.KEEP)
        assertEquals(1f, norm(out), 1e-4f)
        assertTrue(dot(out, g) > dot(e, g) + 0.1f)
    }
}
