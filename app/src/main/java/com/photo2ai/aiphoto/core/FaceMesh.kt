package com.photo2ai.aiphoto.core

import android.graphics.Bitmap
import android.graphics.Rect
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.facemesh.FaceMeshDetection
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** ML Kit 人脸网格（本机运行，模型随 ML Kit 打在包里）和本地换脸的安卓部分 */
object FaceMesh {
    /** 一张脸：468 个点（x,y 按编号排列，原图坐标）、三角形编号、外轮廓编号、外接框 */
    /** [holes] 依次是两只眼睛和嘴的开口轮廓 */
    class Face(val pts: FloatArray, val tris: IntArray, val oval: IntArray, val holes: List<IntArray>, val box: Rect) {
        /** 神经网络换脸用的 5 个关键点。鼻尖是网格的 1 号点，嘴角是 61、291 号（MediaPipe 的人脸网格编号） */
        fun five() = FaceSwap.fivePoints(pts, holes[0], holes[1], 1, 61, 291)
    }

    class NoFace(message: String) : Exception(message)

    /**
     * 有多张脸取最大的，坐标换算回原图。ML Kit 的人脸网格用近距离模型，会把整张图补成正方形再找脸：竖拍、横拍的照片里
     * 脸会被缩得很小而找不到，所以整张找不到时，再沿长边取上中下（左中右）三个正方形各找一次。
     */
    suspend fun detect(src: Bitmap): Face? {
        val side = minOf(src.width, src.height)
        val crops = ArrayList<Rect>()
        crops += Rect(0, 0, src.width, src.height)
        if (src.width.toFloat() / src.height !in 0.8f..1.25f) {
            for (k in 0..2) {
                if (src.height > src.width) {
                    val y = (src.height - side) * k / 2
                    crops += Rect(0, y, side, y + side)
                } else {
                    val x = (src.width - side) * k / 2
                    crops += Rect(x, 0, x + side, side)
                }
            }
        }
        val detector = FaceMeshDetection.getClient(FaceMeshDetectorOptions.Builder().setUseCase(FaceMeshDetectorOptions.FACE_MESH).build())
        try {
            for (c in crops) {
                val part = if (c.width() == src.width && c.height() == src.height) src else Bitmap.createBitmap(src, c.left, c.top, c.width(), c.height())
                val small = ImageIO.fit(part, 1280)
                val k = part.width.toFloat() / small.width
                val meshes = detector.process(InputImage.fromBitmap(small, 0)).await()
                val m = meshes.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() } ?: continue
                val all = m.allPoints
                val pts = FloatArray(all.size * 2)
                for (p in all) {
                    pts[p.index * 2] = p.position.x * k + c.left
                    pts[p.index * 2 + 1] = p.position.y * k + c.top
                }
                val tris = ArrayList<Int>()
                for (t in m.allTriangles) for (p in t.allPoints) tris += p.index
                fun contour(type: Int) = m.getPoints(type).map { it.index }.toIntArray()
                val oval = contour(com.google.mlkit.vision.facemesh.FaceMesh.FACE_OVAL)
                // 嘴的开口：上唇下沿 + 倒过来的下唇上沿
                val mouth = contour(com.google.mlkit.vision.facemesh.FaceMesh.UPPER_LIP_BOTTOM) +
                    contour(com.google.mlkit.vision.facemesh.FaceMesh.LOWER_LIP_TOP).reversedArray()
                val holes = listOf(
                    contour(com.google.mlkit.vision.facemesh.FaceMesh.LEFT_EYE),
                    contour(com.google.mlkit.vision.facemesh.FaceMesh.RIGHT_EYE),
                    mouth,
                )
                val b = m.boundingBox
                val box = Rect((b.left * k).toInt() + c.left, (b.top * k).toInt() + c.top, (b.right * k).toInt() + c.left, (b.bottom * k).toInt() + c.top)
                return Face(pts, tris.toIntArray(), oval, holes, box)
            }
            return null
        } finally {
            detector.close()
        }
    }

    /** 目标脸周围（外扩 [k] 倍脸框）的一块，坐标都换算到这块里 */
    private class Crop(val x0: Int, val y0: Int, val w: Int, val h: Int, val px: IntArray, val pts: FloatArray)

    private fun crop(target: Bitmap, face: Face, k: Float): Crop {
        val b = face.box
        val mx = (b.width() * k).roundToInt()
        val my = (b.height() * k).roundToInt()
        val x0 = max(0, b.left - mx)
        val y0 = max(0, b.top - my)
        val cw = min(target.width, b.right + mx) - x0
        val ch = min(target.height, b.bottom + my) - y0
        if (cw < 8 || ch < 8) throw NoFace("脸太小了，换一张脸大一点的照片")
        val px = IntArray(cw * ch).also { target.getPixels(it, 0, cw, x0, y0, cw, ch) }
        val pts = FloatArray(face.pts.size) { if (it % 2 == 0) face.pts[it] - x0 else face.pts[it] - y0 }
        return Crop(x0, y0, cw, ch, px, pts)
    }

    private fun pasted(target: Bitmap, c: Crop, px: IntArray): Bitmap {
        val out = target.copy(Bitmap.Config.ARGB_8888, true)
        out.setPixels(px, 0, c.w, c.x0, c.y0, c.w, c.h)
        return out
    }

    private suspend fun detectBoth(target: Bitmap, ref: Bitmap): Pair<Face, Face> {
        val dst = detect(target) ?: throw NoFace("当前照片里没有找到人脸")
        val src = detect(ref) ?: throw NoFace("提供脸的照片里没有找到人脸，换一张正脸清楚的")
        return dst to src
    }

    /**
     * 快速换脸（传统做法）：把 [ref] 里的脸按网格贴到 [target] 的脸上。眼睛、嘴两张脸张开程度差不多就一起换，
     * 差得多就保留目标自己的。只在目标脸周围（外扩 25%）的一块里算。
     */
    suspend fun swap(target: Bitmap, ref: Bitmap): Bitmap {
        val (dstFace, srcFace) = detectBoth(target, ref)
        return withContext(Dispatchers.Default) {
            val c = crop(target, dstFace, 0.25f)
            val keep = dstFace.holes.indices.filter { i ->
                !FaceSwap.similarOpenness(FaceSwap.openness(dstFace.pts, dstFace.holes[i]), FaceSwap.openness(srcFace.pts, srcFace.holes[i]))
            }.map { dstFace.holes[it] }
            val swapped = FaceSwap.swap(ImageIO.pixels(ref), ref.width, ref.height, srcFace.pts, c.px, c.w, c.h, c.pts, dstFace.tris, dstFace.oval, keep)
            pasted(target, c, swapped)
        }
    }

    /**
     * 高清换脸（神经网络）：ArcFace 取参考脸的身份，inswapper 换到目标脸上，Real-ESRGAN 放大到 512 变清楚，
     * 按脸部轮廓贴回。两个大模型先后加载、用完就释放，同时占用的内存不超过一个。
     */
    suspend fun neuralSwap(target: Bitmap, ref: Bitmap, arcface: java.io.File, inswapper: java.io.File, esrgan: java.io.File): Bitmap {
        val (dstFace, srcFace) = detectBoth(target, ref)
        return withContext(Dispatchers.Default) {
            val emb = try {
                NeuralSwap.embedding(arcface, ImageIO.pixels(ref), ref.width, ref.height, srcFace.five())
            } finally {
                LocalAi.release(arcface)
            }
            val c = crop(target, dstFace, 0.6f)
            val five = FloatArray(10) { if (it % 2 == 0) dstFace.five()[it] - c.x0 else dstFace.five()[it] - c.y0 }
            // 贴回只用对齐空间里的羽化方框（和 FaceFusion 默认一样）：模型对脸以外的地方基本原样输出，
            // 再按脸部轮廓裁会在下巴、胡子上切出一道断口
            val swapped = try {
                NeuralSwap.swap(inswapper, NeuralSwap.emapOf(inswapper), esrgan, c.px, c.w, c.h, five, emb, null)
            } finally {
                LocalAi.release(inswapper)
            }
            pasted(target, c, swapped)
        }
    }

    private suspend fun <T> Task<T>.await(): T = suspendCancellableCoroutine { cont ->
        addOnSuccessListener { cont.resume(it) }
        addOnFailureListener { cont.resumeWithException(it) }
        addOnCanceledListener { cont.cancel() }
    }
}
