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
    class Face(val pts: FloatArray, val tris: IntArray, val oval: IntArray, val holes: List<IntArray>, val box: Rect)

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

    /**
     * 把 [ref] 里的脸换到 [target] 的脸上。只在目标脸周围（外扩 25%）的一块里算，再贴回去。
     */
    suspend fun swap(target: Bitmap, ref: Bitmap): Bitmap {
        val dstFace = detect(target) ?: throw NoFace("当前照片里没有找到人脸")
        val srcFace = detect(ref) ?: throw NoFace("提供脸的照片里没有找到人脸，换一张正脸清楚的")
        return withContext(Dispatchers.Default) {
            val b = dstFace.box
            val mx = (b.width() * 0.25f).roundToInt()
            val my = (b.height() * 0.25f).roundToInt()
            val x0 = max(0, b.left - mx)
            val y0 = max(0, b.top - my)
            val x1 = min(target.width, b.right + mx)
            val y1 = min(target.height, b.bottom + my)
            val cw = x1 - x0
            val ch = y1 - y0
            if (cw < 8 || ch < 8) throw NoFace("脸太小了，换一张脸大一点的照片")
            val crop = IntArray(cw * ch).also { target.getPixels(it, 0, cw, x0, y0, cw, ch) }
            val dstPts = FloatArray(dstFace.pts.size) { if (it % 2 == 0) dstFace.pts[it] - x0 else dstFace.pts[it] - y0 }
            val srcPx = ImageIO.pixels(ref)
            val swapped = FaceSwap.swap(srcPx, ref.width, ref.height, srcFace.pts, crop, cw, ch, dstPts, dstFace.tris, dstFace.oval, dstFace.holes)
            val out = target.copy(Bitmap.Config.ARGB_8888, true)
            out.setPixels(swapped, 0, cw, x0, y0, cw, ch)
            out
        }
    }

    private suspend fun <T> Task<T>.await(): T = suspendCancellableCoroutine { cont ->
        addOnSuccessListener { cont.resume(it) }
        addOnFailureListener { cont.resumeWithException(it) }
        addOnCanceledListener { cont.cancel() }
    }
}
