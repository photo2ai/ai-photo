package com.photo2ai.aiphoto.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapShader
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.RectF
import android.graphics.Shader
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import kotlin.math.abs
import kotlin.math.min

/**
 * 看图 + 涂抹 + 裁剪框。
 * 缩放平移是作用在视图坐标上的 [zoom]，和图片本身的尺寸无关：换一张同比例的图（比如预览图换成处理结果）时画面不跳。
 */
class PhotoView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    enum class Mode { VIEW, BRUSH, CROP }

    private val density = resources.displayMetrics.density
    private var bmp: Bitmap? = null
    private val zoom = Matrix()
    private val fit = RectF()
    private val drawMatrix = Matrix()
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    private val checker = Paint().apply {
        val tile = Bitmap.createBitmap(2, 2, Bitmap.Config.ARGB_8888)
        tile.setPixel(0, 0, 0xFF3A3A3A.toInt()); tile.setPixel(1, 1, 0xFF3A3A3A.toInt())
        tile.setPixel(1, 0, 0xFF2A2A2A.toInt()); tile.setPixel(0, 1, 0xFF2A2A2A.toInt())
        shader = BitmapShader(tile, Shader.TileMode.REPEAT, Shader.TileMode.REPEAT).apply {
            setLocalMatrix(Matrix().apply { setScale(8 * density, 8 * density) })
        }
    }

    /** 按住对比时显示的图（打开时的样子） */
    var compare: Bitmap? = null
        set(v) {
            field = v
            invalidate()
        }

    var mode = Mode.VIEW
        set(v) {
            field = v
            zoom.reset()
            stroke = null
            computeFit()
            invalidate()
        }

    val image get() = bmp

    fun setImage(b: Bitmap?) {
        val old = bmp
        bmp = b
        if (old == null || b == null || abs(old.width.toFloat() / old.height - b.width.toFloat() / b.height) > 0.01f) zoom.reset()
        computeFit()
        invalidate()
    }

    private fun computeFit() {
        val b = bmp ?: return
        if (width == 0 || height == 0) return
        val pad = if (mode == Mode.CROP) 20 * density else 0f
        val vw = width - 2 * pad
        val vh = height - 2 * pad
        val s = min(vw / b.width, vh / b.height)
        val dw = b.width * s
        val dh = b.height * s
        fit.set((width - dw) / 2, (height - dh) / 2, (width + dw) / 2, (height + dh) / 2)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        zoom.reset()
        computeFit()
    }

    /** 图片像素 → 视图坐标 */
    private fun matrixFor(bw: Int, bh: Int, out: Matrix = drawMatrix): Matrix {
        out.setRectToRect(RectF(0f, 0f, bw.toFloat(), bh.toFloat()), fit, Matrix.ScaleToFit.FILL)
        out.postConcat(zoom)
        return out
    }

    private fun zoomScale(): Float {
        val v = FloatArray(9)
        zoom.getValues(v)
        return v[Matrix.MSCALE_X]
    }

    override fun onDraw(canvas: Canvas) {
        val cmp = compare
        val b = cmp ?: bmp ?: return
        val m = if (cmp != null && bmp != null && abs(cmp.width.toFloat() / cmp.height - bmp!!.width.toFloat() / bmp!!.height) > 0.01f) {
            // 裁剪过，比例不同：对比图单独居中放
            val s = min(width.toFloat() / cmp.width, height.toFloat() / cmp.height)
            Matrix().apply { setScale(s, s); postTranslate((width - cmp.width * s) / 2, (height - cmp.height * s) / 2) }
        } else matrixFor(b.width, b.height)
        if (b.hasAlpha()) {
            val r = RectF(0f, 0f, b.width.toFloat(), b.height.toFloat())
            m.mapRect(r)
            canvas.drawRect(r, checker)
        }
        canvas.drawBitmap(b, m, bitmapPaint)
        if (cmp != null) return
        when (mode) {
            Mode.BRUSH -> drawBrush(canvas)
            Mode.CROP -> drawCrop(canvas)
            Mode.VIEW -> Unit
        }
    }

    // ---------------------------------------------------------------- 手势

    private var lastFocusX = 0f
    private var lastFocusY = 0f
    private var lastCount = 0
    private var multiTouch = false

    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            if (mode == Mode.CROP) return false
            val cur = zoomScale()
            val target = (cur * d.scaleFactor).coerceIn(1f, 12f)
            val f = target / cur
            zoom.postScale(f, f, d.focusX, d.focusY)
            clampPan()
            invalidate()
            return true
        }
    })

    private val gestureDetector = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDoubleTap(e: MotionEvent): Boolean {
            if (mode != Mode.VIEW) return false
            if (zoomScale() > 1.05f) zoom.reset() else zoom.postScale(2.5f, 2.5f, e.x, e.y)
            clampPan()
            invalidate()
            return true
        }
    })

    private fun clampPan() {
        val r = RectF(fit)
        zoom.mapRect(r)
        var dx = 0f
        var dy = 0f
        if (r.width() <= width) dx = width / 2f - r.centerX() else {
            if (r.left > 0) dx = -r.left
            if (r.right < width) dx = width - r.right
        }
        if (r.height() <= height) dy = height / 2f - r.centerY() else {
            if (r.top > 0) dy = -r.top
            if (r.bottom < height) dy = height - r.bottom
        }
        zoom.postTranslate(dx, dy)
    }

    override fun onTouchEvent(e: MotionEvent): Boolean {
        if (bmp == null) return false
        if (mode == Mode.CROP) return cropTouch(e)
        scaleDetector.onTouchEvent(e)
        gestureDetector.onTouchEvent(e)
        var fx = 0f
        var fy = 0f
        val skip = if (e.actionMasked == MotionEvent.ACTION_POINTER_UP) e.actionIndex else -1
        var n = 0
        for (i in 0 until e.pointerCount) {
            if (i == skip) continue
            fx += e.getX(i); fy += e.getY(i); n++
        }
        if (n > 0) { fx /= n; fy /= n }
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                multiTouch = false
                if (mode == Mode.BRUSH) startStroke(e.x, e.y)
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                multiTouch = true
                if (mode == Mode.BRUSH) cancelStroke()
            }
            MotionEvent.ACTION_MOVE -> {
                if (n != lastCount) {
                    lastFocusX = fx; lastFocusY = fy
                } else if (multiTouch || mode == Mode.VIEW) {
                    zoom.postTranslate(fx - lastFocusX, fy - lastFocusY)
                    clampPan()
                    invalidate()
                } else if (mode == Mode.BRUSH) {
                    extendStroke(e.x, e.y)
                }
            }
            MotionEvent.ACTION_UP -> {
                if (mode == Mode.BRUSH && !multiTouch) endStroke()
                brushCursor = false
                invalidate()
            }
            MotionEvent.ACTION_CANCEL -> {
                cancelStroke()
                brushCursor = false
                invalidate()
            }
        }
        lastFocusX = fx
        lastFocusY = fy
        lastCount = if (e.actionMasked == MotionEvent.ACTION_UP || e.actionMasked == MotionEvent.ACTION_CANCEL) 0 else n
        return true
    }

    // ---------------------------------------------------------------- 涂抹

    /** 笔刷半径，视图像素 */
    var brushRadius = 22 * density
    var onMaskChanged: (() -> Unit)? = null
    private var mask: Bitmap? = null
    private var maskCanvas: Canvas? = null
    private val strokes = ArrayList<Pair<Path, Float>>()
    private var stroke: Path? = null
    private var strokeWidthPx = 0f
    private var lastBx = 0f
    private var lastBy = 0f
    private var brushCursor = false
    private var cursorX = 0f
    private var cursorY = 0f
    private val maskPaint = Paint(Paint.FILTER_BITMAP_FLAG).apply { color = Color.argb(140, 255, 60, 90) }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        color = Color.WHITE
    }
    private val cursorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * density
        color = Color.WHITE
    }

    /** 蒙版和当前图片同尺寸（预览尺寸） */
    fun resetMask() {
        val b = bmp ?: return
        mask = Bitmap.createBitmap(b.width, b.height, Bitmap.Config.ALPHA_8)
        maskCanvas = Canvas(mask!!)
        strokes.clear()
        stroke = null
        invalidate()
        onMaskChanged?.invoke()
    }

    val hasMask get() = strokes.isNotEmpty()

    /** 当前蒙版（ALPHA_8）；没有涂过时为 null */
    fun maskBitmap(): Bitmap? = if (strokes.isEmpty()) null else mask

    fun undoStroke() {
        if (strokes.isEmpty()) return
        strokes.removeAt(strokes.size - 1)
        val c = maskCanvas ?: return
        c.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        for ((p, w) in strokes) c.drawPath(p, strokePaint.apply { strokeWidth = w })
        invalidate()
        onMaskChanged?.invoke()
    }

    private fun toBitmapPoint(x: Float, y: Float): FloatArray {
        val m = mask ?: return floatArrayOf(x, y)
        val inv = Matrix()
        matrixFor(m.width, m.height, Matrix()).invert(inv)
        return floatArrayOf(x, y).also { inv.mapPoints(it) }
    }

    private fun startStroke(x: Float, y: Float) {
        val m = mask ?: return
        val scale = fit.width() / m.width * zoomScale()
        strokeWidthPx = 2 * brushRadius / scale
        val p = toBitmapPoint(x, y)
        stroke = Path().apply { moveTo(p[0], p[1]); lineTo(p[0] + 0.01f, p[1]) }
        lastBx = p[0]; lastBy = p[1]
        maskCanvas?.drawPath(stroke!!, strokePaint.apply { strokeWidth = strokeWidthPx })
        brushCursor = true
        cursorX = x; cursorY = y
        invalidate()
    }

    private fun extendStroke(x: Float, y: Float) {
        val s = stroke ?: return
        val p = toBitmapPoint(x, y)
        val seg = Path().apply { moveTo(lastBx, lastBy); lineTo(p[0], p[1]) }
        s.lineTo(p[0], p[1])
        maskCanvas?.drawPath(seg, strokePaint.apply { strokeWidth = strokeWidthPx })
        lastBx = p[0]; lastBy = p[1]
        cursorX = x; cursorY = y
        invalidate()
    }

    private fun endStroke() {
        val s = stroke ?: return
        strokes += s to strokeWidthPx
        stroke = null
        onMaskChanged?.invoke()
    }

    /** 第二根手指按下：这一笔作废（是要缩放），把已经画上去的擦掉 */
    private fun cancelStroke() {
        if (stroke == null) return
        stroke = null
        val c = maskCanvas ?: return
        c.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        for ((p, w) in strokes) c.drawPath(p, strokePaint.apply { strokeWidth = w })
        invalidate()
    }

    private fun drawBrush(canvas: Canvas) {
        val m = mask ?: return
        canvas.drawBitmap(m, matrixFor(m.width, m.height, Matrix()), maskPaint)
        if (brushCursor) canvas.drawCircle(cursorX, cursorY, brushRadius, cursorPaint)
    }

    // ---------------------------------------------------------------- 裁剪

    /** 裁剪框，0..1 的相对坐标 */
    val crop = RectF(0f, 0f, 1f, 1f)

    /** 宽高比（像素），0 表示自由 */
    var cropAspect = 0f
        private set
    private var dragHandle = 0
    private var dragX = 0f
    private var dragY = 0f
    private val startCrop = RectF()
    private val dimPaint = Paint().apply { color = Color.argb(150, 0, 0, 0) }
    private val framePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.WHITE
        strokeWidth = 1.5f * density
    }
    private val gridPaint = Paint().apply {
        color = Color.argb(110, 255, 255, 255)
        strokeWidth = 1f
    }
    private val handlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = 4 * density
        strokeCap = Paint.Cap.ROUND
    }

    /** 设定比例，框取图里能放下的最大居中区域 */
    fun setCropAspect(aspect: Float) {
        cropAspect = aspect
        val b = bmp ?: return
        if (aspect <= 0f) {
            crop.set(0f, 0f, 1f, 1f)
        } else {
            val imgAspect = b.width.toFloat() / b.height
            if (aspect > imgAspect) {
                val h = imgAspect / aspect
                crop.set(0f, (1 - h) / 2, 1f, (1 + h) / 2)
            } else {
                val w = aspect / imgAspect
                crop.set((1 - w) / 2, 0f, (1 + w) / 2, 1f)
            }
        }
        invalidate()
    }

    private fun cropViewRect(): RectF = RectF(
        fit.left + crop.left * fit.width(), fit.top + crop.top * fit.height(),
        fit.left + crop.right * fit.width(), fit.top + crop.bottom * fit.height(),
    )

    private fun drawCrop(canvas: Canvas) {
        val r = cropViewRect()
        canvas.drawRect(fit.left, fit.top, fit.right, r.top, dimPaint)
        canvas.drawRect(fit.left, r.bottom, fit.right, fit.bottom, dimPaint)
        canvas.drawRect(fit.left, r.top, r.left, r.bottom, dimPaint)
        canvas.drawRect(r.right, r.top, fit.right, r.bottom, dimPaint)
        for (i in 1..2) {
            val x = r.left + r.width() * i / 3
            val y = r.top + r.height() * i / 3
            canvas.drawLine(x, r.top, x, r.bottom, gridPaint)
            canvas.drawLine(r.left, y, r.right, y, gridPaint)
        }
        canvas.drawRect(r, framePaint)
        val l = 16 * density
        for ((cx, cy) in listOf(r.left to r.top, r.right to r.top, r.left to r.bottom, r.right to r.bottom)) {
            val sx = if (cx == r.left) 1 else -1
            val sy = if (cy == r.top) 1 else -1
            canvas.drawLine(cx, cy, cx + sx * l, cy, handlePaint)
            canvas.drawLine(cx, cy, cx, cy + sy * l, handlePaint)
        }
    }

    // 拖动的部位：位 1 左、2 上、4 右、8 下；16 整体移动
    private fun cropTouch(e: MotionEvent): Boolean {
        val b = bmp ?: return false
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                val r = cropViewRect()
                val t = 28 * density
                var h = 0
                if (abs(e.x - r.left) < t && e.y > r.top - t && e.y < r.bottom + t) h = h or 1
                if (abs(e.x - r.right) < t && e.y > r.top - t && e.y < r.bottom + t) h = h or 4
                if (abs(e.y - r.top) < t && e.x > r.left - t && e.x < r.right + t) h = h or 2
                if (abs(e.y - r.bottom) < t && e.x > r.left - t && e.x < r.right + t) h = h or 8
                // 框很窄时左右（上下）两条边都在范围内：取离得近的那条
                if (h and 5 == 5) h = h and (if (abs(e.x - r.left) < abs(e.x - r.right)) 4.inv() else 1.inv())
                if (h and 10 == 10) h = h and (if (abs(e.y - r.top) < abs(e.y - r.bottom)) 8.inv() else 2.inv())
                // 锁定比例时只能拖四个角
                if (cropAspect > 0 && h != 0 && Integer.bitCount(h) == 1) h = 0
                if (h == 0 && r.contains(e.x, e.y)) h = 16
                dragHandle = h
                dragX = e.x; dragY = e.y
                startCrop.set(crop)
                return h != 0
            }
            MotionEvent.ACTION_MOVE -> {
                if (dragHandle == 0) return false
                val dx = (e.x - dragX) / fit.width()
                val dy = (e.y - dragY) / fit.height()
                val minW = 48 * density / fit.width()
                val minH = 48 * density / fit.height()
                val c = RectF(startCrop)
                if (dragHandle == 16) {
                    val mx = dx.coerceIn(-c.left, 1 - c.right)
                    val my = dy.coerceIn(-c.top, 1 - c.bottom)
                    c.offset(mx, my)
                } else {
                    if (dragHandle and 1 != 0) c.left = (c.left + dx).coerceIn(0f, c.right - minW)
                    if (dragHandle and 4 != 0) c.right = (c.right + dx).coerceIn(c.left + minW, 1f)
                    if (dragHandle and 2 != 0) c.top = (c.top + dy).coerceIn(0f, c.bottom - minH)
                    if (dragHandle and 8 != 0) c.bottom = (c.bottom + dy).coerceIn(c.top + minH, 1f)
                    if (cropAspect > 0) lockAspect(c, b)
                }
                crop.set(c)
                invalidate()
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> dragHandle = 0
        }
        return true
    }

    /** 拖角时保持比例：以对角为锚点，按宽度定高度，超出边界就整体缩小 */
    private fun lockAspect(c: RectF, b: Bitmap) {
        val ratio = cropAspect * b.height / b.width // 相对坐标里的宽 / 高
        val anchorX = if (dragHandle and 1 != 0) c.right else c.left
        val anchorY = if (dragHandle and 2 != 0) c.bottom else c.top
        var w = c.width()
        var h = w / ratio
        if (h < c.height()) { h = c.height(); w = h * ratio }
        val maxW = if (dragHandle and 1 != 0) anchorX else 1 - anchorX
        val maxH = if (dragHandle and 2 != 0) anchorY else 1 - anchorY
        val s = min(1f, min(maxW / w, maxH / h))
        w *= s; h *= s
        if (dragHandle and 1 != 0) { c.left = anchorX - w; c.right = anchorX } else { c.left = anchorX; c.right = anchorX + w }
        if (dragHandle and 2 != 0) { c.top = anchorY - h; c.bottom = anchorY } else { c.top = anchorY; c.bottom = anchorY + h }
    }
}
