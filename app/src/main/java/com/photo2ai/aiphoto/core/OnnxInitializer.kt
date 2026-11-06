package com.photo2ai.aiphoto.core

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 从 ONNX 模型文件里读出最后一个 initializer（权重张量），转成 float。按 protobuf 的字段长度跳着读，不把整个文件读进内存：
 * ModelProto 的第 7 号字段是 graph，GraphProto 的第 5 号字段是 initializer（TensorProto：1 dims、2 data_type、9 raw_data）。
 */
object OnnxInitializer {
    fun last(file: File, expect: Int): FloatArray {
        RandomAccessFile(file, "r").use { f ->
            var graph: LongArray? = null
            scan(f, 0, f.length()) { field, wire, start, len -> if (field == 7 && wire == 2) graph = longArrayOf(start, len) }
            val g = graph ?: error("模型文件里没有 graph")
            var init: LongArray? = null
            scan(f, g[0], g[0] + g[1]) { field, wire, start, len -> if (field == 5 && wire == 2) init = longArrayOf(start, len) }
            val t = init ?: error("模型文件里没有 initializer")
            var dtype = 0L
            var raw: LongArray? = null
            scan(f, t[0], t[0] + t[1]) { field, wire, start, len ->
                if (field == 2 && wire == 0) dtype = len
                if (field == 9 && wire == 2) raw = longArrayOf(start, len)
            }
            val r = raw ?: error("initializer 没有 raw_data")
            val bytes = ByteArray(r[1].toInt())
            f.seek(r[0])
            f.readFully(bytes)
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val out = when (dtype) {
                1L -> FloatArray(bytes.size / 4) { bb.float }
                10L -> FloatArray(bytes.size / 2) { halfToFloat(bb.short) }
                else -> error("不支持的数据类型 $dtype")
            }
            if (out.size != expect) error("initializer 大小不对：${out.size}")
            return out
        }
    }

    /**
     * 依次读 [from, to) 里的字段，回调（字段号、类型、起点、长度）。varint 字段的「长度」位置放的是它的值。
     */
    private fun scan(f: RandomAccessFile, from: Long, to: Long, onField: (Int, Int, Long, Long) -> Unit) {
        var pos = from
        while (pos < to) {
            f.seek(pos)
            val (key, p1) = varint(f, pos)
            val field = (key ushr 3).toInt()
            val wire = (key and 7).toInt()
            pos = p1
            when (wire) {
                0 -> {
                    val (v, p2) = varint(f, pos)
                    onField(field, wire, pos, v)
                    pos = p2
                }
                1 -> pos += 8
                2 -> {
                    val (len, p2) = varint(f, pos)
                    onField(field, wire, p2, len)
                    pos = p2 + len
                }
                5 -> pos += 4
                else -> error("模型文件格式不对（wire type $wire）")
            }
        }
    }

    private fun varint(f: RandomAccessFile, at: Long): Pair<Long, Long> {
        f.seek(at)
        var result = 0L
        var shift = 0
        var pos = at
        while (true) {
            val b = f.read()
            if (b < 0) error("模型文件不完整")
            pos++
            result = result or ((b and 0x7f).toLong() shl shift)
            if (b < 0x80) return result to pos
            shift += 7
        }
    }

    private fun halfToFloat(h: Short): Float {
        val bits = h.toInt() and 0xffff
        val sign = bits ushr 15
        val exp = (bits ushr 10) and 0x1f
        val mant = bits and 0x3ff
        val v = when (exp) {
            0 -> mant / 1024f * Math.pow(2.0, -14.0).toFloat()
            31 -> if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
            else -> (1 + mant / 1024f) * Math.pow(2.0, (exp - 15).toDouble()).toFloat()
        }
        return if (sign == 1) -v else v
    }
}
