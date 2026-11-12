package com.photo2ai.aiphoto.core

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.FloatBuffer
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs
import kotlin.math.cbrt
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * 本地换装（贴图式，不用生成模型）：把平铺的衣服照片拆成躯干和袖子，按人的姿态变形贴到身上，再处理遮挡、光照、露出来的旧衣服。
 * 衣服的图案、文字完全保留；看起来不如云端生成的自然（没有褶皱）。电脑上的原型和效果对比见 vton-eval 的 tools/sticker_tryon2.py。
 *
 * 用到的模型：ML Kit 姿态（随包）、MODNet 人像抠图、MediaPipe 多类别人像分割（头发 / 皮肤 / 衣服 / 配饰）、
 * Depth Anything V2 Small（深度，给衣服加明暗）、MKL-Harmonizer（调色）、MI-GAN（补掉露出来的旧衣服）。
 */
object TryOn {
    class Models(val modnet: File, val migan: File, val parsing: File, val depth: File, val harmonizer: File)

    class NoPerson(message: String) : Exception(message)

    /** 各步耗时和原生内存，写到 logcat（tag AIPhoto），排查慢、占内存时用 */
    private class Clock {
        private var t = System.nanoTime()
        fun mark(step: String) {
            val now = System.nanoTime()
            android.util.Log.i("AIPhoto", "tryon %s %d ms, native %d MB".format(step, (now - t) / 1_000_000, android.os.Debug.getNativeHeapAllocatedSize() shr 20))
            t = now
        }
    }

    /** 在长边 1024 的图上算（原型的尺寸），最后只把变了的部分放大贴回原图 */
    private const val WORK = 1024

    // ---------------------------------------------------------------- 模型

    private suspend fun <T> Task<T>.await(): T = suspendCancellableCoroutine { cont ->
        addOnSuccessListener { cont.resume(it) }
        addOnFailureListener { cont.resumeWithException(it) }
    }

    /** ML Kit 姿态：33 个点（编号和 MediaPipe BlazePose 一致），返回 (x, y, 在画面里的可能性) */
    private suspend fun pose(src: Bitmap): Array<FloatArray>? {
        val detector = PoseDetection.getClient(
            AccuratePoseDetectorOptions.Builder().setDetectorMode(AccuratePoseDetectorOptions.SINGLE_IMAGE_MODE).build(),
        )
        try {
            val p = detector.process(InputImage.fromBitmap(src, 0)).await()
            val all = p.allPoseLandmarks
            if (all.size < 33) return null
            val out = Array(33) { FloatArray(3) }
            for (l in all) out[l.landmarkType] = floatArrayOf(l.position.x, l.position.y, l.inFrameLikelihood)
            return out
        } finally {
            detector.close()
        }
    }

    /** 多类别人像分割（MediaPipe selfie_multiclass_256x256 转的 ONNX，权重 INT8）：输入 NHWC [-1,1]，输出 6 类 logits */
    class Parts(val hair: FloatArray, val body: FloatArray, val face: FloatArray, val clothes: FloatArray, val others: FloatArray)

    private fun parts(model: File, src: Bitmap, gray: FloatArray): Parts {
        val w = src.width
        val h = src.height
        val n = 256
        val small = ImageIO.pixels(shrink(src, n, n))
        val input = FloatArray(n * n * 3)
        for (i in 0 until n * n) {
            val c = small[i]
            input[i * 3] = (c shr 16 and 0xff) / 127.5f - 1f
            input[i * 3 + 1] = (c shr 8 and 0xff) / 127.5f - 1f
            input[i * 3 + 2] = (c and 0xff) / 127.5f - 1f
        }
        val s = LocalAi.session(model)
        val logits = OnnxTensor.createTensor(LocalAi.env, FloatBuffer.wrap(input), longArrayOf(1, n.toLong(), n.toLong(), 3)).use { t ->
            s.run(mapOf(s.inputNames.first() to t)).use { r ->
                val fb = (r.get(0) as OnnxTensor).floatBuffer
                FloatArray(fb.remaining()).also { fb.get(it) }
            }
        }
        val ch = Array(6) { FloatArray(n * n) }
        for (i in 0 until n * n) {
            var mx = -Float.MAX_VALUE
            for (k in 0 until 6) mx = max(mx, logits[i * 6 + k])
            var sum = 0f
            for (k in 0 until 6) { val e = exp(logits[i * 6 + k] - mx); ch[k][i] = e; sum += e }
            for (k in 0 until 6) ch[k][i] /= sum
        }
        // 放大到工作尺寸，再用导向滤波把边缘贴到原图的边缘上
        val r = max(2, w / 256)
        fun up(k: Int) = Cutout.guided(gray, Pixels.resizeF(ch[k], n, n, w, h), w, h, r, 1e-3f)
        return Parts(up(1), up(2), up(3), up(4), up(5))
    }

    /** Depth Anything V2 Small（INT8）：相对深度（越大越近），放大到工作尺寸 */
    private fun depth(model: File, src: Bitmap): FloatArray {
        val w = src.width
        val h = src.height
        val k = 518f / max(w, h)
        val rw = max(14, (w * k / 14).roundToInt() * 14)
        val rh = max(14, (h * k / 14).roundToInt() * 14)
        val small = ImageIO.pixels(shrink(src, rw, rh))
        val n = rw * rh
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val input = FloatArray(3 * n)
        for (i in 0 until n) {
            val c = small[i]
            input[i] = ((c shr 16 and 0xff) / 255f - mean[0]) / std[0]
            input[n + i] = ((c shr 8 and 0xff) / 255f - mean[1]) / std[1]
            input[2 * n + i] = ((c and 0xff) / 255f - mean[2]) / std[2]
        }
        val s = LocalAi.session(model)
        val d = OnnxTensor.createTensor(LocalAi.env, FloatBuffer.wrap(input), longArrayOf(1, 3, rh.toLong(), rw.toLong())).use { t ->
            s.run(mapOf(s.inputNames.first() to t)).use { r ->
                val fb = (r.get(0) as OnnxTensor).floatBuffer
                FloatArray(fb.remaining()).also { fb.get(it) }
            }
        }
        LocalAi.release(model) // 2500 万参数，用完就放，别和后面的 MI-GAN 一起占着内存
        return Pixels.resizeF(d, rw, rh, w, h)
    }

    /** MKL-Harmonizer：RGB + 遮罩（256×256，0..1）→ 3×3 颜色矩阵（行优先）+ 偏移，作用在 0..1 的行向量 rgb 上 */
    private fun harmonizer(model: File, rgb: FloatArray, mask: FloatArray, w: Int, h: Int): FloatArray {
        val n = 256
        val input = FloatArray(4 * n * n)
        for (c in 0 until 3) {
            val plane = FloatArray(w * h) { rgb[it * 3 + c] }
            System.arraycopy(areaResize(plane, w, h, n, n), 0, input, c * n * n, n * n)
        }
        System.arraycopy(areaResize(mask, w, h, n, n), 0, input, 3 * n * n, n * n)
        val s = LocalAi.session(model)
        return OnnxTensor.createTensor(LocalAi.env, FloatBuffer.wrap(input), longArrayOf(1, 4, n.toLong(), n.toLong())).use { t ->
            s.run(mapOf(s.inputNames.first() to t)).use { r ->
                val fb = (r.get(0) as OnnxTensor).floatBuffer
                FloatArray(fb.remaining()).also { fb.get(it) }
            }
        }
    }

    /** 缩小 Bitmap：每次减半（双线性）直到不到目标的两倍，再缩到目标尺寸，接近面积平均，不会有锯齿 */
    private fun shrink(b: Bitmap, dw: Int, dh: Int): Bitmap {
        var cur = b
        while (cur.width / 2 >= dw && cur.height / 2 >= dh) cur = Bitmap.createScaledBitmap(cur, cur.width / 2, cur.height / 2, true)
        return if (cur.width == dw && cur.height == dh) cur else Bitmap.createScaledBitmap(cur, dw, dh, true)
    }

    /** 缩小：先按缩放比例模糊再双线性取样（近似 INTER_AREA） */
    private fun areaResize(v: FloatArray, w: Int, h: Int, dw: Int, dh: Int): FloatArray {
        val r = max(w / dw, h / dh) / 2
        return Pixels.resizeF(if (r >= 1) Pixels.blurF(v, w, h, r, 1) else v, w, h, dw, dh)
    }

    // ---------------------------------------------------------------- 主流程

    /** [ann] 不为 null 时是内置衣服：按人工标注拆衣服、贴体（见 [Garment.Annotation]），否则自动识别（只认上衣） */
    suspend fun run(src: Bitmap, cloth: Bitmap, m: Models, ann: Garment.Annotation? = null): Bitmap {
        val work = if (max(src.width, src.height) > WORK) withContext(Dispatchers.Default) { ImageIO.fit(src, WORK) } else src
        val clock = Clock()
        val lm = pose(work) ?: throw NoPerson("没找到人。请用能看到上半身（肩膀到腰）的正面照片")
        // 只要求两肩在画面里；腰、手在画面外时 ML Kit 也会给出推测的位置，够用
        for (k in intArrayOf(11, 12)) if (lm[k][2] < 0.5f) throw NoPerson("没看清两边肩膀，请用能看到上半身的正面照片")
        clock.mark("pose")
        val result = withContext(Dispatchers.Default) { render(work, cloth, lm, m, clock, ann) }
        return withContext(Dispatchers.Default) { pasteBack(src, work, result).also { clock.mark("pasteBack") } }
    }

    /** 只把变了的地方放大贴回原图：脸、背景保持原图清晰度 */
    private fun pasteBack(src: Bitmap, work: Bitmap, result: IntArray): Bitmap {
        val w = work.width
        val h = work.height
        if (src.width == w && src.height == h) return ImageIO.toBitmap(result, w, h)
        val orig = ImageIO.pixels(work)
        var changed = BooleanArray(orig.size) { orig[it] != result[it] }
        changed = Masks.dilate(changed, w, h, 2)
        val soft = Pixels.blurF(FloatArray(changed.size) { if (changed[it]) 1f else 0f }, w, h, 1, 2)
        val sw = src.width
        val sh = src.height
        val big = ImageIO.pixels(ImageIO.scaled(ImageIO.toBitmap(result, w, h), sw, sh))
        val mask = Pixels.resizeF(soft, w, h, sw, sh)
        return ImageIO.toBitmap(Pixels.composite(big, ImageIO.pixels(src), mask), sw, sh)
    }

    private fun render(work: Bitmap, clothBmp: Bitmap, lm: Array<FloatArray>, m: Models, clock: Clock, ann: Garment.Annotation?): IntArray {
        val w = work.width
        val h = work.height
        val n = w * h
        val px = ImageIO.pixels(work)
        val gray = Pixels.lumaF(px)
        val P = { k: Int -> V(lm[k][0], lm[k][1]) }

        val alpha = Matting.person(m.modnet, work).resized(w, h)
        LocalAi.release(m.modnet) // MODNet 的会话要占约 450 MB，后面还要跑深度和 MI-GAN，用完就放
        val person = BooleanArray(n) { alpha[it] > 0.5f }
        clock.mark("modnet")
        val parts = parts(m.parsing, work, gray)
        LocalAi.release(m.parsing)
        val skin = FloatArray(n) { min(1f, parts.body[it] + parts.face[it]) * alpha[it] }
        clock.mark("parsing")
        val dep = depth(m.depth, work)
        clock.mark("depth")

        // 画面左边的肩 = x 小的那个；对应的肘、腕、手用同一侧的编号
        val (iL, iR) = if (lm[11][0] < lm[12][0]) 11 to 12 else 12 to 11
        val pl = P(iL)
        val pr = P(iR)
        val sw = (pr - pl).norm()
        val across = (pr - pl) / sw
        var up = across.rot90().let { if (it.y < 0) it else -it }
        if ((P(0) - (pl + pr) / 2f) dot up < 0) up = -up // 指向头

        // 衣服
        val cw = clothBmp.width
        val chh = clothBmp.height
        val cloth = ImageIO.pixels(clothBmp)
        // 内置衣服是抠好的透明 WebP，直接用透明度当蒙版（细肩带、蕾丝边经不起开运算）
        val cmask = if (ann != null) BooleanArray(cloth.size) { (cloth[it] ushr 24) >= 128 } else Garment.cutout(cloth, cw, chh)
        val g = if (ann != null) Garment.annotated(cmask, cw, chh, ann) else Garment.parse(cmask, cw, chh)
        clock.mark("garment")

        // 肩部目标：从肩关节沿「向上」走到人像轮廓，再往外一点（衣服肩线在关节外上方）
        fun shoulderTop(p: V): Float {
            var d = 0.04f
            while (d < 0.3f) {
                val q = p + up * (d * sw)
                val qx = q.x.toInt()
                val qy = q.y.toInt()
                if (qx !in 0 until w || qy !in 0 until h || alpha[qy * w + qx] < 0.5f) return d * sw
                d += 0.01f
            }
            return 0.1f * sw
        }
        val tsl = pl - across * (0.06f * sw) + up * max(shoulderTop(pl) - 0.01f * sw, 0.06f * sw)
        val tsr = pr + across * (0.06f * sw) + up * max(shoulderTop(pr) - 0.01f * sw, 0.06f * sw)
        // 衣服上的向量 → 人身上：旋转 + 缩放（让衣服两肩对上人的两肩）
        val gv = g.sr - g.sl
        val pv = tsr - tsl
        // 落肩的袍服：衣服上 shoulder 这么宽对应人的肩宽，竖向（衣长、袖长）按这个比例；肩点仍对人的肩，横向下面 fit 收拢
        val s = pv.norm() / (g.shoulder ?: gv.norm())
        val ang = kotlin.math.atan2(pv.y, pv.x) - kotlin.math.atan2(gv.y, gv.x)
        val ca = kotlin.math.cos(ang) * s
        val sa = kotlin.math.sin(ang) * s
        val anchorG = if (g.shoulder != null) (g.sl + g.sr) / 2f else g.sl
        val anchorP = if (g.shoulder != null) (tsl + tsr) / 2f else tsl
        fun f(p: V): V { val d = p - anchorG; return anchorP + V(ca * d.x - sa * d.y, sa * d.x + ca * d.y) }
        var tal = f(g.al)
        var tar = f(g.ar)
        var thl = f(g.hl)
        var thr = f(g.hr)
        val tn = f(g.neck)

        // 身体比衣服宽时把衣服撑开（只放大不缩小）；垂在两边的手臂不算身体
        val arms = BooleanArray(n)
        for (k in intArrayOf(iL, iR)) Masks.capsule(w, h, listOf(P(k), P(k + 2), P(k + 4)), 0.13f * sw, arms)
        val body = BooleanArray(n) { person[it] && !arms[it] }
        /** a、b 中点所在那一行身体连续的一段 (左, 右)；中点不在身体上返回 null */
        fun bodySpan(a: V, b: V): Pair<Int, Int>? {
            val y = ((a.y + b.y) / 2).toInt().coerceIn(0, h - 1)
            val mx = ((a.x + b.x) / 2).toInt().coerceIn(0, w - 1)
            if (!body[y * w + mx]) return null
            var l = mx
            while (l > 0 && body[y * w + l - 1]) l--
            var r = mx
            while (r < w - 1 && body[y * w + r + 1]) r++
            return l to r
        }
        /** 以身体中心把 a、b 缩放 k 倍 */
        fun scaleAt(a: V, b: V, span: Pair<Int, Int>, k: Float): Pair<V, V> {
            val mid = (a + b) / 2f
            val c = V((span.first + span.second) / 2f, mid.y)
            return (c + (a - mid) * k) to (c + (b - mid) * k)
        }
        fun widen(a: V, b: V): Pair<V, V> {
            val span = bodySpan(a, b) ?: return a to b
            val need = (span.second - span.first) * 1.04f
            val have = (b - a).norm()
            return if (have >= need) a to b else scaleAt(a, b, span, need / have)
        }
        /** 内置衣服：这一行宽度按身体宽度 × ease 定（可收可放），中心对到身体中心；返回新的两点和缩放比例 */
        fun fit(a: V, b: V, ease: Float): Triple<V, V, Float> {
            val span = bodySpan(a, b)
            if (span == null) {
                // 这一行中点不在身体上（侧身、手臂挡在胸前、抠图断开）：按人的肩宽 × ease 收拢，落肩的袍子不能原样放着（比人宽近一倍）
                val k = (tsr - tsl).norm() * ease / ((b - a).norm() + 1e-6f)
                val mid = (a + b) / 2f
                return Triple(mid + (a - mid) * k, mid + (b - mid) * k, k)
            }
            val k = (span.second - span.first) * ease / ((b - a).norm() + 1e-6f)
            val (a1, b1) = scaleAt(a, b, span, k)
            return Triple(a1, b1, k)
        }
        // 腰（收腰的内置衣服才有）：网格多一行
        var waist: Pair<V, V>? = if (g.wl != null && g.wr != null) f(g.wl) to f(g.wr) else null
        // 落肩的衣服按真实比例放大后一定比人宽，没写 ease 也要收拢
        val ease = g.ease ?: if (g.shoulder != null) 1.4f else null
        if (ease != null) {
            val (a1, b1, kA) = fit(tal, tar, ease)
            tal = a1; tar = b1
            waist = waist?.let { (l, r) -> fit(l, r, ease).let { it.first to it.second } }
            // 腋下收窄了，下摆按同样比例收（保持衣服原来的外扩比例）
            if (kA < 1f) {
                val c = (thl + thr) / 2f
                thl = c + (thl - c) * kA
                thr = c + (thr - c) * kA
            }
        } else {
            widen(tal, tar).let { tal = it.first; tar = it.second }
            waist = waist?.let { (l, r) -> widen(l, r) }
        }
        widen(thl, thr).let { thl = it.first; thr = it.second }

        // 躯干网格：列 = 左外侧 / 左边线 / 中线 / 右边线 / 右外侧，行 = 衣服顶 / 肩 / 腋下 /（腰，内置衣服才有）/ 下摆 / 衣服最底（系带等）。
        // 边线上肩到腋下这段和袖根是同一条直线，袖子接上去没有缝
        val bottom = g.bottom + 2f
        var mg = 0.35f * (g.sr.x - g.sl.x)
        run {
            // 外侧列要盖住所有躯干像素：钟形裙摆、A 字下摆在两行之间鼓出两行连线，固定的余量会把鼓出来的部分切掉
            val ls = listOfNotNull(V(g.sl.x, g.top.toFloat()), g.sl, g.al, g.wl, g.hl, V(g.hl.x, bottom))
            val rs = listOfNotNull(V(g.sr.x, g.top.toFloat()), g.sr, g.ar, g.wr, g.hr, V(g.hr.x, bottom))
            fun sideX(side: List<V>, y: Float): Float {
                if (y <= side.first().y) return side.first().x
                for (i in 0 until side.size - 1) {
                    val a = side[i]
                    val b = side[i + 1]
                    if (y <= b.y) return if (b.y - a.y < 1e-3f) b.x else a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y)
                }
                return side.last().x
            }
            var excess = 0f
            for (y in 0 until chh) {
                val xl = sideX(ls, y.toFloat())
                val xr = sideX(rs, y.toFloat())
                for (x in 0 until cw) if (g.torso[y * cw + x]) excess = max(excess, max(xl - x, x - xr))
            }
            mg = max(mg, excess + 2f)
        }
        val down = -up
        fun row(lg: V, rg: V, lp: V, rp: V, cg: V? = null, cp: V? = null): Pair<List<V>, List<V>> {
            val od = (rp - lp).unit()
            val k = (rp - lp).norm() / ((rg - lg).norm() + 1e-6f)
            return listOf(lg - V(mg, 0f), lg, cg ?: ((lg + rg) / 2f), rg, rg + V(mg, 0f)) to
                listOf(lp - od * (mg * k), lp, cp ?: ((lp + rp) / 2f), rp, rp + od * (mg * k))
        }
        val rows = listOf(
            // 领口点低于两肩（方领、一字领的礼服）时顶行中点比肩行还低，网格会翻折：改用两肩连线中点
            if (g.neck.y <= min(g.sl.y, g.sr.y)) {
                row(V(g.sl.x, g.top.toFloat()), V(g.sr.x, g.top.toFloat()), tsl + up * ((g.sl.y - g.top) * s), tsr + up * ((g.sr.y - g.top) * s), g.neck, tn)
            } else {
                row(V(g.sl.x, g.top.toFloat()), V(g.sr.x, g.top.toFloat()), tsl + up * ((g.sl.y - g.top) * s), tsr + up * ((g.sr.y - g.top) * s))
            },
            row(g.sl, g.sr, tsl, tsr),
            row(g.al, g.ar, tal, tar),
        ) + (waist?.let { (l, r) -> listOf(row(g.wl!!, g.wr!!, l, r)) } ?: emptyList()) + listOf(
            row(g.hl, g.hr, thl, thr),
            row(V(g.hl.x, bottom), V(g.hr.x, bottom), thl + down * ((bottom - g.hl.y) * s), thr + down * ((bottom - g.hr.y) * s)),
        )
        val torsoW = warpTris(garmentBitmap(cloth, g.torso, cw, chh), gridPairs(rows.map { it.first }, rows.map { it.second }), w, h)

        // 袖子：沿 肩 → 肘 → 腕 切成一串四边形贴上
        class SleeveInfo(val root: V, val len: Float, val long: Boolean)
        val sleeveInfo = HashMap<Int, SleeveInfo>()
        val sleevesW = ArrayList<IntArray>()
        for ((side, sl) in g.sleeves) {
            val sG = if (side == 'L') g.sl else g.sr
            val aG = if (side == 'L') g.al else g.ar
            val sP = if (side == 'L') tsl else tsr
            val aP = if (side == 'L') tal else tar
            val k = if (side == 'L') iL else iR
            val root = (sP + aP) / 2f
            val el = P(k + 2)
            val wr = P(k + 4)
            val ext = wr + (wr - el).unit() * sw // 腕往外延长，长袖可以盖过手腕
            val upper = (el - root).norm()
            // 折线圆角（Chaikin 3 次）：肘部弯得急（自拍时手在胸前）时，截面方向要逐步转过去，不然外侧四边形会被拉成扇面甩出去
            var poly = listOf(root, el, wr, ext)
            repeat(3) {
                val q = arrayListOf(poly.first())
                for ((a, b) in poly.zipWithNext()) { q += a * 0.75f + b * 0.25f; q += a * 0.25f + b * 0.75f }
                q += poly.last()
                poly = q
            }
            val seg = poly.zipWithNext { a, b -> (b - a).norm() }
            fun at(len: Float): Pair<V, V> {
                var l = len
                for (i in seg.indices) {
                    if (l <= seg[i] || i == seg.size - 1) {
                        val u = (poly[i + 1] - poly[i]) / (seg[i] + 1e-6f)
                        return (poly[i] + u * l) to u
                    }
                    l -= seg[i]
                }
                error("unreachable")
            }
            val ax = sl.axis
            val ng = ax.rot90()
            // 袖子横截面上哪一侧是肩（外缘）：两个依据取更可靠的那个。「腋下→肩」在袖子水平伸出时可靠、手臂下垂时几乎和袖轴平行；
            // 「衣身在哪一侧」正好相反。只用前者，垂袖会随机翻面扭成细条甩出去；只用后者，T 字形平铺的袖子会翻面
            fun shoulderSide(sh: V, arm: V, root0: V, center: V, n: V): Int {
                val a = ((sh - arm) dot n) / ((sh - arm).norm() + 1e-6f)
                val b = ((center - root0) dot n) / ((center - root0).norm() + 1e-6f)
                return if (abs(a) >= abs(b)) sign(a) else -sign(b)
            }
            val sideG = shoulderSide(sG, aG, sl.root, V(g.cx.toFloat(), (sG.y + aG.y) / 2), ng)
            val sideP = shoulderSide(sP, aP, root, (tsl + tsr) / 2f, at(1f).second.rot90())
            val flip = sideG != sideP
            val sgnS = sideG.toFloat()
            val nSt = if (sl.length > 1.25f * upper / s) 12 else 3
            // 截面垂直于袖轴，袖根却是竖的：第一个截面要在袖根两端之后，否则四边形会扭成交叉
            val tFirst = max(max((sG - sl.root) dot ax, (aG - sl.root) dot ax), 0f) + 0.08f * sl.length
            val ts = if (tFirst < 0.9f * sl.length) List(nSt) { tFirst + (1.03f * sl.length - tFirst) * it / max(nSt - 1, 1) } else listOf(1.03f * sl.length)
            // 袖根往躯干里多伸一点：避免和躯干之间露缝
            val e = 0.06f * (sG - aG).norm()
            val u0 = (poly[1] - poly[0]).unit()
            val gEdges = arrayListOf((sG - ax * e) to (aG - ax * e))
            val pEdges = arrayListOf((sP - u0 * (e * s)) to (aP - u0 * (e * s)))
            // 袖子像素在轴向 / 横向的坐标，用来量每个截面的宽度
            val cnt = Masks.count(sl.mask)
            val tt = FloatArray(cnt)
            val dd = FloatArray(cnt)
            var j = 0
            for (i in sl.mask.indices) if (sl.mask[i]) {
                val d = V((i % cw).toFloat(), (i / cw).toFloat()) - sl.root
                tt[j] = d dot ax; dd[j] = d dot ng; j++
            }
            val tMax = tt.max()
            for (tc in ts) {
                var lo = Float.MAX_VALUE
                var hi = -Float.MAX_VALUE
                fun scan(t0: Float, band: Float) {
                    for (i in tt.indices) if (abs(tt[i] - t0) < band) { lo = min(lo, dd[i]); hi = max(hi, dd[i]) }
                }
                scan(tc, max(3f, 0.06f * sl.length))
                if (lo > hi) scan(min(tc, tMax), max(3f, 0.08f * sl.length))
                if (lo > hi) continue
                val pad = 0.1f * (hi - lo)
                lo -= pad; hi += pad
                val (dS, dA) = if (sgnS > 0) hi to lo else lo to hi
                gEdges += (sl.root + ax * tc + ng * dS) to (sl.root + ax * tc + ng * dA)
                val (p, u) = at(tc * s)
                val nrm = u.rot90() * ((if (flip) -1f else 1f) * s * g.sleeveSqueeze)
                pEdges += (p + nrm * dS) to (p + nrm * dA)
            }
            val ext2 = Masks.dilate(sl.mask, cw, chh, e.toInt() + 2)
            val extMask = BooleanArray(sl.mask.size) { sl.mask[it] || (ext2[it] && g.mask[it]) }
            val pairs = ArrayList<Pair<List<V>, List<V>>>()
            for (i in 0 until gEdges.size - 1) {
                val (g0, g1) = gEdges[i]
                val (q0, q1) = gEdges[i + 1]
                val (p0, p1) = pEdges[i]
                val (r0, r1) = pEdges[i + 1]
                pairs += listOf(g0, g1, q1) to listOf(p0, p1, r1)
                pairs += listOf(g0, q1, q0) to listOf(p0, r1, r0)
            }
            sleevesW += warpTris(garmentBitmap(cloth, extMask, cw, chh), pairs, w, h)
            sleeveInfo[k] = SleeveInfo(root, sl.length * s, sl.length * s > 1.1f * upper)
        }

        clock.mark("warp")
        return compose(px, w, h, alpha, person, skin, parts, dep, lm, iL, iR, sw, arms, torsoW, sleevesW, sleeveInfo.mapValues { Triple(it.value.root, it.value.len, it.value.long) }, m, clock, g.long)
    }

    private fun sign(v: Float) = if (v >= 0) 1 else -1

    /** 衣服图 + 蒙版 → 带透明度的 Bitmap（蒙版外透明） */
    private fun garmentBitmap(cloth: IntArray, mask: BooleanArray, w: Int, h: Int): Bitmap =
        ImageIO.toBitmap(IntArray(cloth.size) { if (mask[it]) cloth[it] or Pixels.ALPHA else 0 }, w, h)

    /** 网格顶点 [行][列] → 每个格子拆成两个三角形 */
    private fun gridPairs(gr: List<List<V>>, pr: List<List<V>>): List<Pair<List<V>, List<V>>> {
        val out = ArrayList<Pair<List<V>, List<V>>>()
        for (i in 0 until gr.size - 1) for (j in 0 until gr[0].size - 1) {
            val idx = listOf(i to j, i to j + 1, i + 1 to j + 1, i + 1 to j)
            for (t in listOf(listOf(0, 1, 2), listOf(0, 2, 3))) {
                out += t.map { gr[idx[it].first][idx[it].second] } to t.map { pr[idx[it].first][idx[it].second] }
            }
        }
        return out
    }

    /** 逐个三角形仿射：衣服上三点 → 人身上三点，裁到目标三角形（稍放大一点免得露缝） */
    private fun warpTris(src: Bitmap, pairs: List<Pair<List<V>, List<V>>>, w: Int, h: Int): IntArray {
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        val mtx = Matrix()
        for ((st, dt) in pairs) {
            val area = (dt[1].x - dt[0].x) * (dt[2].y - dt[0].y) - (dt[1].y - dt[0].y) * (dt[2].x - dt[0].x)
            if (abs(area) < 1f) continue // 退化三角形
            val sp = floatArrayOf(st[0].x, st[0].y, st[1].x, st[1].y, st[2].x, st[2].y)
            val dp = floatArrayOf(dt[0].x, dt[0].y, dt[1].x, dt[1].y, dt[2].x, dt[2].y)
            if (!mtx.setPolyToPoly(sp, 0, dp, 0, 3)) continue
            val c = (dt[0] + dt[1] + dt[2]) / 3f
            val path = Path()
            dt.map { c + (it - c) * 1.03f }.forEachIndexed { i, p -> if (i == 0) path.moveTo(p.x, p.y) else path.lineTo(p.x, p.y) }
            path.close()
            canvas.save()
            canvas.clipPath(path)
            canvas.drawBitmap(src, mtx, paint)
            canvas.restore()
        }
        val px = ImageIO.pixels(out)
        out.recycle()
        return px
    }

    // ---------------------------------------------------------------- 合成

    private fun compose(
        px: IntArray, w: Int, h: Int, alpha: FloatArray, person: BooleanArray, skin: FloatArray, parts: Parts, dep: FloatArray,
        lm: Array<FloatArray>, iL: Int, iR: Int, sw: Float, arms: BooleanArray,
        torsoW: IntArray, sleevesW: List<IntArray>, sleeveInfo: Map<Int, Triple<V, Float, Boolean>>, m: Models, clock: Clock,
        long: Boolean,
    ): IntArray {
        val n = w * h
        val P = { k: Int -> V(lm[k][0], lm[k][1]) }
        val layers = (listOf(torsoW) + sleevesW).map { Layer.of(it) }
        val covA = FloatArray(n)
        for (l in layers) for (i in 0 until n) covA[i] = max(covA[i], l.a[i])
        val cover = BooleanArray(n) { covA[it] > 0.5f }

        // ---- 光照：曝光按脸的亮度（衣服照片是棚拍白光），空间明暗取原图衣服区域的大尺度亮度，再按深度法线让身体侧面暗一点
        val lab = Color.labOf(px)
        val lum = FloatArray(n) { lab[it * 3] / 100f }
        val face = BooleanArray(n)
        Masks.capsule(w, h, listOf(P(0)), max(3f, 0.12f * sw), face)
        for (i in 0 until n) face[i] = face[i] && skin[i] > 0.5f
        val expo = if (Masks.count(face) > 20) (Masks.median(lum, face) / 0.68f).coerceIn(0.45f, 1.1f) else 1f
        val hipY = (lm[23][1] + lm[24][1]) / 2
        // 光照只按原来的衣服估：手、手机、头发、皮肤的亮度不是衣服上的光照。长款只用胯部以上（下面是裤子、腿）
        val oldCloth = BooleanArray(n) { person[it] && parts.clothes[it] > 0.5f && skin[it] < 0.3f && parts.hair[it] < 0.5f }
        val region = BooleanArray(n) { cover[it] && oldCloth[it] && (!long || it / w < hipY) }
        val light = lighting(lum, region, w, h, sw, long)
        val inside = Masks.erode(BooleanArray(n) { cover[it] && person[it] }, w, h, max(2, (0.04f * sw).toInt()))
        if (Masks.count(inside) > 100) {
            // 只在人身上平滑深度：人和背景之间的深度落差模糊开，会在身体边缘一整圈压暗（领口、肩上的灰环）
            val sr = Masks.gaussianSigmaToBox(max(1.5f, 0.03f * sw))
            val pw = Pixels.blurF(FloatArray(n) { if (person[it]) 1f else 0f }, w, h, sr)
            val pd = Pixels.blurF(FloatArray(n) { if (person[it]) dep[it] else 0f }, w, h, sr)
            val d = FloatArray(n) { pd[it] / max(pw[it], 1e-3f) }
            val gx = FloatArray(n)
            val gy = FloatArray(n)
            for (y in 0 until h) for (x in 0 until w) {
                val i = y * w + x
                // 人体外没有深度依据（d 在那里掉到 0，梯度很陡）：不加明暗，免得衣服超出身体的那一圈被压暗
                if (pw[i] < 0.5f) continue
                gx[i] = (d[y * w + min(x + 1, w - 1)] - d[y * w + max(x - 1, 0)]) / (if (x in 1 until w - 1) 2f else 1f)
                gy[i] = (d[min(y + 1, h - 1) * w + x] - d[max(y - 1, 0) * w + x]) / (if (y in 1 until h - 1) 2f else 1f)
            }
            val gm = FloatArray(Masks.count(inside))
            var k = 0
            for (i in 0 until n) if (inside[i]) gm[k++] = sqrt(gx[i] * gx[i] + gy[i] * gy[i])
            val scale = 1.2f / (Masks.percentile(gm, 90f) + 1e-6f)
            val shD = FloatArray(n) {
                val nz = 1f / sqrt(1f + (scale * gx[it]).pow(2) + (scale * gy[it]).pow(2))
                nz.coerceIn(0.5f, 1f).pow(0.6f)
            }
            val med = Masks.median(shD, inside)
            for (i in 0 until n) {
                var k = (shD[i] / med).coerceIn(if (long) 0.8f else 0.6f, if (long) 1.05f else 1.08f)
                // 长款的深度明暗只用在胯部以上：裙摆不贴腿，两腿之间的深度变化不是裙子的起伏
                if (long) k = 1 + (k - 1) * ((hipY + 0.2f * sw - i / w) / (0.2f * sw)).coerceIn(0f, 1f)
                light[i] *= k
            }
        }
        // 白平衡：原图整体偏色（灰世界）的一半迁移到衣服上
        var mr = 0.0; var mgr = 0.0; var mb = 0.0
        for (c in px) { mr += c shr 16 and 0xff; mgr += c shr 8 and 0xff; mb += c and 0xff }
        val mm = (mr + mgr + mb) / 3 + 1e-3
        val cast = doubleArrayOf(mr, mgr, mb).map { (1 + 0.5 * (it / mm - 1)).toFloat().coerceIn(0.85f, 1.15f) }
        val folds = folds(px, w, h, sw, BooleanArray(n) { cover[it] && oldCloth[it] && it / w < hipY + 0.1f * sw })
        val e22 = expo.pow(2.2f)
        for (l in layers) for (i in 0 until n) if (l.a[i] > 0f) {
            val lt = light[i].pow(2.2f) * e22 * folds[i]
            for (c in 0 until 3) l.rgb[i * 3 + c] = (l.rgb[i * 3 + c].pow(2.2f) * lt * cast[c]).coerceIn(0f, 1f).pow(1 / 2.2f)
        }

        clock.mark("light")
        val orig = FloatArray(n * 3) { ((px[it / 3] shr (16 - 8 * (it % 3))) and 0xff) / 255f }
        // 衣服在身上投下的淡阴影：衣服遮罩模糊后往下挪一点，压暗衣服外面的像素
        val base = orig.copyOf()
        val sh = Pixels.blurF(covA, w, h, Masks.gaussianSigmaToBox(max(1f, 0.025f * sw)))
        val shift = (0.015f * sw).roundToInt()
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            val v = if (y - shift >= 0) sh[(y - shift) * w + x] else 0f
            val k = 1 - 0.3f * v * (1 - covA[i])
            for (c in 0 until 3) base[i * 3 + c] *= k
        }

        // ---- 调色：先合成一次给 MKL-Harmonizer 看，预测颜色矩阵；只取它亮度变化的 60%、色度变化的 25%（衣服本身的颜色不能变）
        run {
            val tmp = base.copyOf()
            for (l in layers) over(tmp, l, w, h)
            val f = harmonizer(m.harmonizer, tmp, covA, w, h)
            LocalAi.release(m.harmonizer)
            for (l in layers) for (i in 0 until n) if (l.a[i] > 0f) {
                val r = l.rgb[i * 3]; val gg = l.rgb[i * 3 + 1]; val b = l.rgb[i * 3 + 2]
                val r1 = (r * f[0] + gg * f[3] + b * f[6] + f[9]).coerceIn(0f, 1f)
                val g1 = (r * f[1] + gg * f[4] + b * f[7] + f[10]).coerceIn(0f, 1f)
                val b1 = (r * f[2] + gg * f[5] + b * f[8] + f[11]).coerceIn(0f, 1f)
                val l0 = Color.rgbToLab(r, gg, b)
                val l1 = Color.rgbToLab(r1, g1, b1)
                val mix = Color.labToRgb(l0[0] + (l1[0] - l0[0]) * 0.6f, l0[1] + (l1[1] - l0[1]) * 0.25f, l0[2] + (l1[2] - l0[2]) * 0.25f)
                for (c in 0 until 3) l.rgb[i * 3 + c] = mix[c].coerceIn(0f, 1f)
            }
        }

        clock.mark("harmonize")
        // ---- 遮挡
        val yy = { i: Int -> (i / w).toFloat() }
        val shY = (lm[11][1] + lm[12][1]) / 2
        // 原来的上衣：分割里的「衣服」类，再排除颜色更像裤子的（裤子也算衣服类）
        val clothes = BooleanArray(n) { parts.clothes[it] > 0.5f }
        val topS = BooleanArray(n) { clothes[it] && yy(it) > shY && yy(it) < (shY + hipY) / 2 && !arms[it] }
        val botS = BooleanArray(n) { clothes[it] && yy(it) > hipY + 0.3f * sw }
        val oldTop = clothes.copyOf()
        if (Masks.count(topS) > 100 && Masks.count(botS) > 100) {
            val mt = FloatArray(3) { c -> Masks.median(FloatArray(n) { lab[it * 3 + c] }, topS) }
            val mbb = FloatArray(3) { c -> Masks.median(FloatArray(n) { lab[it * 3 + c] }, botS) }
            fun dist(i: Int, ref: FloatArray) = sqrt((0 until 3).sumOf { c -> ((lab[i * 3 + c] - ref[c]) * (lab[i * 3 + c] - ref[c])).toDouble() })
            if (sqrt((0 until 3).sumOf { ((mt[it] - mbb[it]) * (mt[it] - mbb[it])).toDouble() }) > 15) {
                for (i in 0 until n) if (oldTop[i] && dist(i, mt) >= dist(i, mbb)) oldTop[i] = false
            }
        }
        // 手臂：上臂被袖子盖住的那段不贴回（否则肩上会露出一条皮肤）；短袖时前臂在袖子上面（抬手自拍），长袖时在下面
        val upperSkin = BooleanArray(n)
        val foreUnder = BooleanArray(n)
        val foreOver = BooleanArray(n)
        for (k in intArrayOf(iL, iR)) {
            val shp = P(k)
            val el = P(k + 2)
            val wr = P(k + 4)
            val upM = Masks.capsule(w, h, listOf(shp, el), 0.13f * sw)
            val foM = Masks.capsule(w, h, listOf(el, wr), 0.12f * sw)
            val info = sleeveInfo[k]
            val u = info?.let { (el - it.first).unit() }
            for (i in 0 until n) {
                val skinOk = person[i] && skin[i] > 0.45f
                if (upM[i] && skinOk) {
                    val keep = info == null || ((V((i % w).toFloat(), yy(i)) - info.first) dot u!!) > info.second * 0.9f
                    if (keep) upperSkin[i] = true
                }
                if (foM[i] && skinOk) if (info != null && info.third) foreUnder[i] = true else foreOver[i] = true
            }
        }
        fun smooth(mk: BooleanArray): FloatArray {
            val o = Masks.open(mk, w, h, 1)
            return Pixels.blurF(FloatArray(n) { if (o[it]) 1f else 0f }, w, h, 1)
        }
        // 手和手里的东西：每只手的关键点凸包；两只手靠得近（捧着手机自拍）时两只手之间也算
        val handPts = mapOf(iL to listOf(iL + 4, iL + 6, iL + 8, iL + 10).map(P), iR to listOf(iR + 4, iR + 6, iR + 8, iR + 10).map(P))
        var hands = BooleanArray(n)
        for (k in intArrayOf(iL, iR)) Masks.fillConvex(w, h, Masks.convexHull(handPts[k]!!), hands)
        if ((P(iL + 4) - P(iR + 4)).norm() < 0.8f * sw) Masks.fillConvex(w, h, Masks.convexHull(handPts[iL]!! + handPts[iR]!!), hands)
        hands = Masks.dilate(hands, w, h, max(1, (0.07f * sw).toInt()))
        val held = Masks.open(BooleanArray(n) { hands[it] && person[it] && (skin[it] > 0.25f || parts.others[it] > 0.4f || parts.clothes[it] < 0.5f) }, w, h, 1)
        val front = Pixels.blurF(FloatArray(n) { if (held[it]) 1f else 0f }, w, h, 1)
        // 头和脖子：下巴以上的原图盖回去，避免领子盖到脸
        val mouthY = (lm[9][1] + lm[10][1]) / 2
        val chinY = mouthY + (mouthY - (lm[2][1] + lm[5][1]) / 2) * 0.8f
        val head = Pixels.blurF(FloatArray(n) { if (person[it] && yy(it) < chinY) 1f else 0f }, w, h, 1)
        // 头发在衣服前面（披到肩上的长发、短发发梢）
        val hairM = Pixels.blurF(FloatArray(n) { ((parts.hair[it] - 0.3f) / 0.4f).coerceIn(0f, 1f) * alpha[it] }, w, h, 1)

        val res = base
        fun paste(mk: FloatArray) { for (i in 0 until n) if (mk[i] > 0f) for (c in 0 until 3) res[i * 3 + c] += (orig[i * 3 + c] - res[i * 3 + c]) * mk[i] }
        over(res, layers[0], w, h)
        paste(smooth(BooleanArray(n) { upperSkin[it] || foreUnder[it] }))
        for (l in layers.drop(1)) over(res, l, w, h)
        paste(smooth(foreOver))
        paste(front)
        paste(hairM)
        paste(head)

        var out = IntArray(n) {
            Pixels.ALPHA or (Pixels.clamp255(res[it * 3] * 255f) shl 16) or (Pixels.clamp255(res[it * 3 + 1] * 255f) shl 8) or Pixels.clamp255(res[it * 3 + 2] * 255f)
        }
        clock.mark("occlusion")
        // 新衣服没盖住、露出来的旧上衣：用 MI-GAN 补掉（和「消除」同一个模型）。长款在整件衣服的高度内都补，裤子也算
        val coverAny = BooleanArray(n) { covA[it] > 0.3f }
        val bandEnd = if (long) {
            var i = n - 1
            while (i > 0 && !coverAny[i]) i--
            (i / w).toFloat()
        } else {
            hipY + 0.15f * sw
        }
        val old = if (long) clothes else oldTop
        val armsEnd = hipY + 0.15f * sw
        var shownOld = BooleanArray(n) {
            val y = yy(it)
            person[it] && !coverAny[it] && y > chinY && y < bandEnd && old[it] && skin[it] < 0.3f &&
                front[it] <= 0.3f && !hands[it] && parts.hair[it] < 0.5f &&
                // 长款往下补到裙摆，但胯部以下垂着的手臂（穿着长袖）不是要去掉的旧衣服
                !(y >= armsEnd && arms[it])
        }
        shownOld = Masks.open(shownOld, w, h, 2)
        shownOld = Masks.dropSmall(shownOld, w, h, (0.0005f * n).toInt())
        if (Masks.count(shownOld) > 0) {
            val grown = Masks.dilate(shownOld, w, h, 4)
            val hole = FloatArray(n) { if (grown[it] && !coverAny[it]) 1f else 0f }
            out = LocalAi.inpaint(m.migan, out, w, h, hole)
        }
        clock.mark("inpaint")
        return out
    }

    /**
     * 原图大尺度明暗：模糊亮度 / 区域中位亮度，限制在 0.7～1.15；离 region（原来的衣服）远的地方没有依据，按 1。
     * [long]（长款，region 只有胯部以上）：幅度收窄到 0.85～1.1
     */
    private fun lighting(lum: FloatArray, region: BooleanArray, w: Int, h: Int, sw: Float, long: Boolean): FloatArray {
        val n = w * h
        val r = Masks.gaussianSigmaToBox(max(2f, sw * 0.12f))
        val wsum = Pixels.blurF(FloatArray(n) { if (region[it]) 1f else 0f }, w, h, r)
        val lb = Pixels.blurF(FloatArray(n) { if (region[it]) lum[it] else 0f }, w, h, r)
        val med = if (Masks.count(region) > 0) Masks.median(lum, region) else 0.5f
        return FloatArray(n) {
            val v = 1 + ((lb[it] / max(wsum[it], 1e-3f) / max(med, 1e-3f)).coerceIn(0.7f, 1.15f) - 1) * (wsum[it] / 0.3f).coerceIn(0f, 1f)
            if (long) v.coerceIn(0.85f, 1.1f) else v
        }
    }

    /**
     * 褶皱迁移：原图里旧衣服 [region] 的明暗细节（褶皱、身体起伏、照片颗粒）除以它的局部均值，得到乘在新衣服上的比例（线性亮度）。
     * 旧衣服的图案不能带过来：印字、口袋缝线、牛仔布纹是锐利边缘，褶皱是柔和过渡——锐利边缘附近不迁移，
     * 边缘密集的地方（印花、粗纹理、暗处噪点）整片不迁移。电脑原型上对比过（vton-eval 的原型副本 tryon3.py）
     */
    private fun folds(px: IntArray, w: Int, h: Int, sw: Float, region: BooleanArray): FloatArray {
        val n = w * h
        val out = FloatArray(n) { 1f }
        if (Masks.count(region) < 200) return out
        val lin = FloatArray(256) { (it / 255f).pow(2.2f) }
        val y = FloatArray(n) { val c = px[it]; 0.2126f * lin[c shr 16 and 0xff] + 0.7152f * lin[c shr 8 and 0xff] + 0.0722f * lin[c and 0xff] }
        val r = Masks.gaussianSigmaToBox(max(2f, 0.05f * sw))
        val rw = Pixels.blurF(FloatArray(n) { if (region[it]) 1f else 0f }, w, h, r)
        val ry = Pixels.blurF(FloatArray(n) { if (region[it]) y[it] else 0f }, w, h, r)
        val feather = Pixels.blurF(FloatArray(n).also { f -> Masks.erode(region, w, h, 1).forEachIndexed { i, b -> if (b) f[i] = 1f } }, w, h, Masks.gaussianSigmaToBox(1.5f))
        // 边缘：对数亮度轻微模糊后的梯度
        val lg = Pixels.blurF(FloatArray(n) { kotlin.math.ln(y[it] + 0.01f) }, w, h, 1)
        val edge = BooleanArray(n)
        for (yy in 1 until h - 1) for (x in 1 until w - 1) {
            val i = yy * w + x
            val gx = (lg[i + 1] - lg[i - 1]) / 2
            val gy = (lg[i + w] - lg[i - w]) / 2
            edge[i] = gx * gx + gy * gy > 0.07f * 0.07f
        }
        val near = Pixels.blurF(FloatArray(n).also { f -> Masks.dilate(edge, w, h, 3).forEachIndexed { i, b -> if (b) f[i] = 1f } }, w, h, Masks.gaussianSigmaToBox(1.5f))
        val dens = Pixels.blurF(FloatArray(n) { if (edge[it]) 1f else 0f }, w, h, Masks.gaussianSigmaToBox(max(2f, 0.03f * sw)))
        for (i in 0 until n) {
            if (feather[i] <= 0f) continue
            val base = ry[i] / max(rw[i], 1e-3f)
            // 压缩大的偏差（口袋、印花边缘），保留柔和的明暗；很暗的旧衣服信噪比低，少用
            val d = 1 + kotlin.math.tanh((y[i] / max(base, 1e-3f) - 1) / 0.35f) * 0.35f
            val k = ((base - 0.01f) / 0.04f).coerceIn(0f, 1f)
            val wgt = (1 - near[i]) * (1 - (dens[i] - 0.05f) / 0.1f).coerceIn(0f, 1f)
            out[i] = 1 + (d - 1) * k * feather[i] * wgt * 0.9f
        }
        return out
    }

    /** 一层衣服：rgb（0..1，每像素 3 个）+ 透明度 */
    private class Layer(val rgb: FloatArray, val a: FloatArray) {
        companion object {
            fun of(px: IntArray) = Layer(
                FloatArray(px.size * 3) { ((px[it / 3] shr (16 - 8 * (it % 3))) and 0xff) / 255f },
                FloatArray(px.size) { (px[it] ushr 24) / 255f },
            )
        }
    }

    /** 把一层叠到 [base] 上：透明度先轻微模糊，边缘柔一点 */
    private fun over(base: FloatArray, l: Layer, w: Int, h: Int) {
        val a = Pixels.blurF(l.a, w, h, 1, 1)
        for (i in a.indices) {
            val k = a[i]
            if (k <= 0f) continue
            for (c in 0 until 3) base[i * 3 + c] += (l.rgb[i * 3 + c] - base[i * 3 + c]) * k
        }
    }

    /** sRGB ↔ CIE Lab（D65，L 0..100），和 OpenCV 浮点版的 COLOR_RGB2LAB 一致 */
    object Color {
        private fun lin(c: Float) = if (c <= 0.04045f) c / 12.92f else ((c + 0.055f) / 1.055f).pow(2.4f)
        private fun gam(c: Float) = if (c <= 0.0031308f) 12.92f * c else 1.055f * c.pow(1 / 2.4f) - 0.055f
        private fun fl(t: Float) = if (t > 0.008856f) cbrt(t) else 7.787f * t + 16f / 116f
        private fun fi(t: Float) = if (t > 0.206893f) t * t * t else (t - 16f / 116f) / 7.787f

        fun rgbToLab(r: Float, g: Float, b: Float): FloatArray {
            val lr = lin(r); val lg = lin(g); val lb = lin(b)
            val x = (0.412453f * lr + 0.357580f * lg + 0.180423f * lb) / 0.950456f
            val y = 0.212671f * lr + 0.715160f * lg + 0.072169f * lb
            val z = (0.019334f * lr + 0.119193f * lg + 0.950227f * lb) / 1.088754f
            val fy = fl(y)
            return floatArrayOf(if (y > 0.008856f) 116f * cbrt(y) - 16f else 903.3f * y, 500f * (fl(x) - fy), 200f * (fy - fl(z)))
        }

        fun labToRgb(l: Float, a: Float, b: Float): FloatArray {
            val fy = (l + 16f) / 116f
            val x = fi(fy + a / 500f) * 0.950456f
            val y = fi(fy)
            val z = fi(fy - b / 200f) * 1.088754f
            val r = 3.240479f * x - 1.537150f * y - 0.498535f * z
            val g = -0.969256f * x + 1.875992f * y + 0.041556f * z
            val bb = 0.055648f * x - 0.204043f * y + 1.057311f * z
            return floatArrayOf(gam(r.coerceIn(0f, 1f)), gam(g.coerceIn(0f, 1f)), gam(bb.coerceIn(0f, 1f)))
        }

        /** 0..255 → 线性亮度的查表（整张图转 Lab 时省掉每个像素三次 pow） */
        private val LIN = FloatArray(256) { lin(it / 255f) }

        fun labOf(px: IntArray): FloatArray {
            val out = FloatArray(px.size * 3)
            for (i in px.indices) {
                val c = px[i]
                val lr = LIN[c shr 16 and 0xff]; val lg = LIN[c shr 8 and 0xff]; val lb = LIN[c and 0xff]
                val x = (0.412453f * lr + 0.357580f * lg + 0.180423f * lb) / 0.950456f
                val y = 0.212671f * lr + 0.715160f * lg + 0.072169f * lb
                val z = (0.019334f * lr + 0.119193f * lg + 0.950227f * lb) / 1.088754f
                val fy = fl(y)
                out[i * 3] = if (y > 0.008856f) 116f * fy - 16f else 903.3f * y
                out[i * 3 + 1] = 500f * (fl(x) - fy)
                out[i * 3 + 2] = 200f * (fy - fl(z))
            }
            return out
        }
    }
}
