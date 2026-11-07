package com.photo2ai.aiphoto.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class AccelMathTest {
    private val ref = FloatArray(1000) { (it % 100) / 100f }

    @Test
    fun agreesToleratesHalfPrecisionNoise() {
        assertTrue(AccelMath.agrees(ref, ref.copyOf()))
        assertTrue(AccelMath.agrees(ref, FloatArray(ref.size) { ref[it] + if (it % 2 == 0) 0.004f else -0.004f }))
    }

    @Test
    fun rejectsWrongResults() {
        assertFalse(AccelMath.agrees(ref, FloatArray(ref.size)))                          // 全 0
        assertFalse(AccelMath.agrees(ref, FloatArray(ref.size) { if (it < 50) 1f else ref[it] }.also { it[0] = 9f })) // 一处离谱
        assertFalse(AccelMath.agrees(ref, FloatArray(ref.size) { if (it == 3) Float.NaN else ref[it] }))
        assertFalse(AccelMath.agrees(ref, FloatArray(10)))                               // 形状不对
    }

    @Test
    fun picksFasterOnlyWhenClearlyFaster() {
        val b = Accel.Backend.entries
        assertEquals(Accel.Backend.NNAPI, AccelMath.pick(mapOf(b[0] to 100.0, b[1] to 95.0, b[2] to 40.0)))
        assertEquals(Accel.Backend.CPU, AccelMath.pick(mapOf(b[0] to 100.0, b[1] to 95.0)))   // 只快 5%，不换
        assertEquals(Accel.Backend.CPU, AccelMath.pick(emptyMap()))
    }

    @Test
    fun sourceSignalIsUnitVector() {
        val v = AccelMath.signal(512, unitVector = true)
        assertEquals(1.0, v.sumOf { (it * it).toDouble() }, 1e-4)
        assertTrue(AccelMath.signal(100, unitVector = false).all { it in 0f..1f })
    }

    @Test
    fun clampedTileRepeatsEdges() {
        val w = 3
        val h = 2
        val px = IntArray(w * h) { Pixels.rgb(-1, it * 40, 0, 0) }
        val t = Tensors.tileToChwF32Clamped(px, w, h, 1, 0, 4, 3)
        // 第 0 行：x=1,2,2,2；第 2 行（超出）重复最后一行
        val red = FloatArray(12) { t[it] * 255f }
        assertEquals(listOf(40f, 80f, 80f, 80f, 160f, 200f, 200f, 200f, 160f, 200f, 200f, 200f), red.map { kotlin.math.round(it) })
        assertTrue(abs(t[12]) < 1e-6) // 绿色通道
    }
}
