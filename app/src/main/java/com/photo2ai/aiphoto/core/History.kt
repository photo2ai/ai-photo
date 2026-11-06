package com.photo2ai.aiphoto.core

import android.graphics.Bitmap
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 撤销 / 重做。每一步的整图按原始像素存在缓存目录（不压缩，读写快），内存里只留当前这一张。
 */
class History(private val dir: File, private val limit: Int = 20) {
    private val files = ArrayList<File>()
    private var index = -1
    private var seq = 0

    init {
        dir.mkdirs()
        dir.listFiles()?.forEach { it.delete() }
    }

    val canUndo get() = index > 0
    val canRedo get() = index < files.size - 1

    /** 有没有在打开之后改过 */
    val edited get() = index > 0

    fun push(b: Bitmap) {
        while (files.size > index + 1) files.removeAt(files.size - 1).delete()
        val f = File(dir, "h${seq++}.raw")
        write(f, b)
        files += f
        index = files.size - 1
        // 超出上限丢掉最早的，但第 0 张（打开时的样子）一直留着
        while (files.size > limit) {
            files.removeAt(1).delete()
            index--
        }
    }

    fun undo(): Bitmap? = if (canUndo) read(files[--index]) else null
    fun redo(): Bitmap? = if (canRedo) read(files[++index]) else null

    fun clear() {
        files.forEach { it.delete() }
        files.clear()
        index = -1
    }

    private fun write(f: File, b: Bitmap) {
        val buf = ByteBuffer.allocateDirect(b.byteCount + 8).order(ByteOrder.nativeOrder())
        buf.putInt(b.width).putInt(b.height)
        b.copyPixelsToBuffer(buf)
        buf.flip()
        RandomAccessFile(f, "rw").use { raf ->
            raf.setLength(0)
            raf.channel.write(buf)
        }
    }

    private fun read(f: File): Bitmap {
        RandomAccessFile(f, "r").use { raf ->
            val buf = ByteBuffer.allocateDirect(raf.length().toInt()).order(ByteOrder.nativeOrder())
            raf.channel.read(buf)
            buf.flip()
            val w = buf.int
            val h = buf.int
            val b = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            b.copyPixelsFromBuffer(buf.slice())
            return b
        }
    }
}
