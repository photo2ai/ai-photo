package com.photo2ai.aiphoto.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Outline
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.ViewOutlineProvider
import android.view.inputmethod.InputMethodManager
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.IntentCompat
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.slider.LabelFormatter
import com.google.android.material.slider.Slider
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import com.photo2ai.aiphoto.R
import com.photo2ai.aiphoto.core.Adjust
import com.photo2ai.aiphoto.core.Cloud
import com.photo2ai.aiphoto.core.CloudProtocol
import com.photo2ai.aiphoto.core.Cutout
import com.photo2ai.aiphoto.core.Enhance
import com.photo2ai.aiphoto.core.Filters
import com.photo2ai.aiphoto.core.History
import com.photo2ai.aiphoto.core.ImageIO
import com.photo2ai.aiphoto.core.Inpaint
import com.photo2ai.aiphoto.core.Pixels
import com.photo2ai.aiphoto.core.Segment
import com.photo2ai.aiphoto.core.Store
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.abs
import kotlin.math.roundToInt

class EditorActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_TOOL = "tool"
        const val TOOL_ENHANCE = "enhance"
        const val TOOL_AI = "ai"
        const val TOOL_ERASE = "erase"
        const val TOOL_CUTOUT = "cutout"
        const val TOOL_ADJUST = "adjust"
        const val TOOL_FILTER = "filter"
        const val TOOL_CROP = "crop"

        /** 预览图（显示、实时预览都用它）的长边 */
        private const val PREVIEW_SIDE = 2048
    }

    private lateinit var photo: PhotoView
    private lateinit var panel: FrameLayout
    private lateinit var toolPanel: View
    private lateinit var tools: View
    private lateinit var toolTitle: TextView
    private lateinit var busy: View
    private lateinit var busyText: TextView
    private lateinit var busyCancel: View
    private lateinit var info: TextView
    private lateinit var compareBtn: View
    private lateinit var undoBtn: ImageButton
    private lateinit var redoBtn: ImageButton
    private lateinit var shareBtn: ImageButton
    private lateinit var saveBtn: MaterialButton
    private lateinit var toolApply: View
    private lateinit var toolCancel: View

    /** 全尺寸的当前图 */
    private var current: Bitmap? = null

    /** 当前图的预览尺寸版本 */
    private var preview: Bitmap? = null

    /** 打开时的预览图，按住对比用 */
    private var original: Bitmap? = null
    private lateinit var history: History
    private var dirty = false
    private var tool: Tool? = null
    private var busyJob: Job? = null

    private val storagePermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) save() else toast("没有存储权限，无法保存")
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_editor)
        findViewById<View>(R.id.root).applySystemInsets()
        photo = findViewById(R.id.photo)
        panel = findViewById(R.id.panel)
        toolPanel = findViewById(R.id.toolPanel)
        tools = findViewById(R.id.tools)
        toolTitle = findViewById(R.id.toolTitle)
        busy = findViewById(R.id.busy)
        busyText = findViewById(R.id.busyText)
        busyCancel = findViewById(R.id.busyCancel)
        info = findViewById(R.id.info)
        compareBtn = findViewById(R.id.compare)
        undoBtn = findViewById(R.id.undo)
        redoBtn = findViewById(R.id.redo)
        shareBtn = findViewById(R.id.share)
        saveBtn = findViewById(R.id.save)
        toolApply = findViewById(R.id.toolApply)
        toolCancel = findViewById(R.id.toolCancel)

        history = History(File(cacheDir, "history"))
        findViewById<View>(R.id.close).setOnClickListener { onBackPressedDispatcher.onBackPressed() }
        undoBtn.setOnClickListener { step(undo = true) }
        redoBtn.setOnClickListener { step(undo = false) }
        saveBtn.setOnClickListener { save() }
        shareBtn.setOnClickListener { share() }
        toolApply.setOnClickListener { applyTool() }
        toolCancel.setOnClickListener { closeTool() }
        busyCancel.setOnClickListener { busyJob?.cancel() }
        compareBtn.setOnTouchListener { v, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    photo.compare = tool?.basePreview ?: original
                    v.isPressed = true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    photo.compare = null
                    v.isPressed = false
                }
            }
            true
        }
        buildToolBar()
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = back()
        })
        updateButtons()

        val uri = intent.data ?: IntentCompat.getParcelableExtra(intent, Intent.EXTRA_STREAM, Uri::class.java)
        if (uri == null) {
            toast("没有收到图片")
            finish()
            return
        }
        load(uri, intent.getStringExtra(EXTRA_TOOL))
    }

    override fun onResume() {
        super.onResume()
        tool?.onResume()
    }

    private fun load(uri: Uri, startTool: String?) {
        showBusy("正在打开…", false)
        busyJob = lifecycleScope.launch {
            try {
                val b = withContext(Dispatchers.IO) { ImageIO.decode(this@EditorActivity, uri) }
                val p = withContext(Dispatchers.Default) { ImageIO.fit(b, PREVIEW_SIDE) }
                withContext(Dispatchers.IO) { history.push(b) }
                current = b
                preview = p
                original = p
                photo.setImage(p)
                compareBtn.isVisible = true
                flashInfo("${b.width} × ${b.height}")
                hideBusy()
                if (startTool != null) photo.post { openTool(startTool) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                android.util.Log.w("AIPhoto", "load failed: $uri", e)
                hideBusy()
                toast("打不开这张图片：" + (if (e is OutOfMemoryError) "图片太大" else e.message ?: e.javaClass.simpleName), long = true)
                finish()
            }
        }
    }

    // ---------------------------------------------------------------- 忙碌状态

    private fun showBusy(text: String, cancellable: Boolean) {
        busyText.text = text
        busyCancel.isVisible = cancellable
        busy.isVisible = true
        setControlsEnabled(false)
    }

    private fun hideBusy() {
        busy.isVisible = false
        setControlsEnabled(true)
    }

    /** 跑一件耗时的事：期间显示进度、禁用按钮；出错弹窗说明 */
    private fun busy(text: String, cancellable: Boolean = false, block: suspend () -> Unit) {
        if (busyJob?.isActive == true) return
        showBusy(text, cancellable)
        busyJob = lifecycleScope.launch {
            try {
                block()
            } catch (e: CancellationException) {
                throw e
            } catch (e: OutOfMemoryError) {
                alert("内存不足", "这张图太大，处理时内存不够。可以先裁小一些再试。")
            } catch (e: CloudProtocol.Failure) {
                alert("AI 处理失败", e.message ?: "")
            } catch (e: Exception) {
                android.util.Log.w("AIPhoto", "$text failed", e)
                alert("出错了", e.message ?: e.javaClass.simpleName)
            } finally {
                hideBusy()
            }
        }
    }

    private fun setControlsEnabled(on: Boolean) {
        toolApply.isEnabled = on
        toolCancel.isEnabled = on
        toolApply.alpha = if (on) 1f else 0.35f
        updateButtons(on)
    }

    private fun updateButtons(enabled: Boolean = busyJob?.isActive != true) {
        val idle = enabled && tool == null && current != null
        fun set(v: View, on: Boolean) {
            v.isEnabled = on
            v.alpha = if (on) 1f else 0.35f
        }
        set(undoBtn, idle && history.canUndo)
        set(redoBtn, idle && history.canRedo)
        set(saveBtn, idle)
        set(shareBtn, idle)
    }

    private fun alert(title: String, msg: String) {
        if (isFinishing) return
        MaterialAlertDialogBuilder(this).setTitle(title).setMessage(msg).setPositiveButton("好", null).show()
    }

    private val hideInfo = Runnable { info.isVisible = false }

    private fun flashInfo(text: String, ms: Long = 2500) {
        info.text = text
        info.isVisible = true
        info.removeCallbacks(hideInfo)
        if (ms > 0) info.postDelayed(hideInfo, ms)
    }

    // ---------------------------------------------------------------- 撤销、保存、返回

    private fun step(undo: Boolean) {
        busy(if (undo) "正在撤销…" else "正在重做…") {
            val b = withContext(Dispatchers.IO) { if (undo) history.undo() else history.redo() } ?: return@busy
            val p = withContext(Dispatchers.Default) { ImageIO.fit(b, PREVIEW_SIDE) }
            current = b
            preview = p
            dirty = true
            photo.setImage(p)
        }
    }

    /** 确认一个工具的结果：进撤销栈、换成当前图 */
    private suspend fun commit(b: Bitmap) {
        val p = withContext(Dispatchers.Default) { ImageIO.fit(b, PREVIEW_SIDE) }
        withContext(Dispatchers.IO) { history.push(b) }
        current = b
        preview = p
        dirty = true
    }

    private fun save() {
        val b = current ?: return
        if (Build.VERSION.SDK_INT < 29 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
        ) {
            storagePermission.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            return
        }
        busy("正在保存…") {
            val where = withContext(Dispatchers.IO) { ImageIO.saveToGallery(this@EditorActivity, b) }
            dirty = false
            toast("已保存到 $where", long = true)
        }
    }

    private fun share() {
        val b = current ?: return
        busy("正在准备…") {
            val uri = withContext(Dispatchers.IO) { ImageIO.shareUri(this@EditorActivity, b) }
            val mime = if (uri.toString().endsWith(".png")) "image/png" else "image/jpeg"
            startActivity(
                Intent.createChooser(
                    Intent(Intent.ACTION_SEND).setType(mime).putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION),
                    "分享到",
                ),
            )
        }
    }

    private fun back() {
        if (busyJob?.isActive == true) {
            if (busyCancel.isVisible) busyJob?.cancel()
            return
        }
        if (tool != null) {
            closeTool()
            return
        }
        if (!dirty) {
            finish()
            return
        }
        MaterialAlertDialogBuilder(this)
            .setTitle("放弃修改？")
            .setMessage("修改还没有保存到相册。")
            .setPositiveButton("保存") { _, _ -> save() }
            .setNegativeButton("放弃") { _, _ -> finish() }
            .setNeutralButton("继续编辑", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isFinishing) history.clear()
    }

    // ---------------------------------------------------------------- 工具栏

    private class ToolDef(val id: String, val icon: Int, val label: String)

    private val toolDefs = listOf(
        ToolDef(TOOL_ENHANCE, R.drawable.ic_magic, "智能增强"),
        ToolDef(TOOL_AI, R.drawable.ic_sparkle, "AI 修图"),
        ToolDef(TOOL_ERASE, R.drawable.ic_eraser, "消除"),
        ToolDef(TOOL_CUTOUT, R.drawable.ic_person, "抠图"),
        ToolDef(TOOL_ADJUST, R.drawable.ic_tune, "调整"),
        ToolDef(TOOL_FILTER, R.drawable.ic_filter, "滤镜"),
        ToolDef(TOOL_CROP, R.drawable.ic_crop, "裁剪"),
    )

    private fun buildToolBar() {
        val row = findViewById<LinearLayout>(R.id.toolRow)
        for (d in toolDefs) {
            row.addView(LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(dp(4), dp(6), dp(4), dp(6))
                minimumWidth = dp(72)
                background = ContextCompat.getDrawable(context, android.R.drawable.list_selector_background)
                isClickable = true
                setOnClickListener { openTool(d.id) }
                addView(ImageView(context).apply {
                    setImageResource(d.icon)
                    if (d.id == TOOL_AI) setColorFilter(ContextCompat.getColor(context, R.color.accent))
                }, LinearLayout.LayoutParams(dp(26), dp(26)))
                addView(TextView(context).apply {
                    text = d.label
                    textSize = 12f
                    setTextColor(ContextCompat.getColor(context, R.color.text))
                    setPadding(0, dp(6), 0, 0)
                })
            })
        }
    }

    private fun openTool(id: String) {
        if (tool != null || current == null || busyJob?.isActive == true) return
        val t = when (id) {
            TOOL_ENHANCE -> EnhanceTool()
            TOOL_AI -> AiTool()
            TOOL_ERASE -> EraseTool()
            TOOL_CUTOUT -> CutoutTool()
            TOOL_ADJUST -> AdjustTool()
            TOOL_FILTER -> FilterTool()
            TOOL_CROP -> CropTool()
            else -> return
        }
        tool = t
        toolTitle.text = t.title
        panel.removeAllViews()
        panel.addView(t.panel())
        toolPanel.isVisible = true
        tools.isVisible = false
        // 裁剪时对比按钮会压住裁剪框右下角
        compareBtn.isVisible = t !is CropTool
        updateButtons()
        t.enter()
    }

    private fun closeTool() {
        val t = tool ?: return
        t.exit()
        tool = null
        nextRender = null
        photo.mode = PhotoView.Mode.VIEW
        photo.setImage(preview)
        panel.removeAllViews()
        toolPanel.isVisible = false
        tools.isVisible = true
        compareBtn.isVisible = true
        info.isVisible = false
        hideKeyboard()
        updateButtons()
    }

    private fun applyTool() {
        val t = tool ?: return
        busy(t.applyText) {
            val r = t.result()
            if (r != null) commit(r)
            closeTool()
        }
    }

    private fun hideKeyboard() {
        val v = currentFocus ?: return
        getSystemService(InputMethodManager::class.java).hideSoftInputFromWindow(v.windowToken, 0)
        v.clearFocus()
    }

    // ---------------------------------------------------------------- 实时预览

    private var previewTarget: Bitmap? = null
    private var rendering = false
    private var nextRender: (() -> IntArray)? = null

    /** 在预览图上算效果并显示。算的时候又有新请求就只留最新的一个，算完再接着算 */
    private fun render(compute: () -> IntArray) {
        nextRender = compute
        if (!rendering) drain()
    }

    private fun drain() {
        val c = nextRender ?: return
        nextRender = null
        rendering = true
        val owner = tool
        lifecycleScope.launch {
            val px = try {
                withContext(Dispatchers.Default) { c() }
            } catch (e: OutOfMemoryError) {
                null
            }
            rendering = false
            if (px != null && owner != null && tool === owner) showPixels(px)
            drain()
        }
    }

    private fun showPixels(px: IntArray) {
        val p = preview ?: return
        if (px.size != p.width * p.height) return
        var t = previewTarget
        if (t == null || t.width != p.width || t.height != p.height) {
            t = Bitmap.createBitmap(p.width, p.height, Bitmap.Config.ARGB_8888)
            previewTarget = t
        }
        t!!.setPixels(px, 0, p.width, 0, 0, p.width, p.height)
        if (photo.image !== t) photo.setImage(t) else photo.invalidate()
    }

    // ---------------------------------------------------------------- 面板控件

    private fun column(vararg views: View) = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(16), dp(12), dp(16), dp(4))
        for (v in views) addView(v)
    }

    private fun hint(text: String) = TextView(this).apply {
        this.text = text
        textSize = 12.5f
        setTextColor(ContextCompat.getColor(context, R.color.text2))
        setPadding(0, 0, 0, dp(4))
    }

    /** 名称 + 数值 + 滑杆 */
    private fun sliderRow(label: String, from: Float, to: Float, value: Float, format: (Float) -> String = { it.roundToInt().toString() }, onChange: (Float) -> Unit): Pair<View, Slider> {
        val valueText = TextView(this).apply {
            text = format(value)
            textSize = 13f
            setTextColor(ContextCompat.getColor(context, R.color.text))
            minWidth = dp(36)
            gravity = Gravity.END
        }
        val slider = Slider(this).apply {
            valueFrom = from
            valueTo = to
            this.value = value
            stepSize = 1f
            isTickVisible = false
            labelBehavior = LabelFormatter.LABEL_GONE
            addOnChangeListener { _, v, fromUser ->
                valueText.text = format(v)
                if (fromUser) onChange(v)
            }
        }
        val row = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            addView(TextView(context).apply {
                text = label
                textSize = 13f
                setTextColor(ContextCompat.getColor(context, R.color.text2))
                minWidth = dp(40)
            })
            addView(slider, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(valueText)
        }
        return row to slider
    }

    private fun choiceChip(text: String) = Chip(this).apply {
        this.text = text
        isCheckable = true
        isCheckedIconVisible = false
        val accent = ContextCompat.getColor(context, R.color.accent_dark)
        val normal = ContextCompat.getColor(context, R.color.surface2)
        chipBackgroundColor = ColorStateList(arrayOf(intArrayOf(android.R.attr.state_checked), intArrayOf()), intArrayOf(accent, normal))
        chipStrokeWidth = 0f
        setEnsureMinTouchTargetSize(false)
    }

    /**
     * 一行可以横向滑动的单选标签。不用单行 ChipGroup：中英文标签混排时它会把中文标签的底部裁掉一截
     */
    private fun chipRow(labels: List<String>, selected: Int, onSelect: (Int) -> Unit): Pair<View, LinearLayout> {
        val row = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        labels.forEachIndexed { i, s ->
            row.addView(choiceChip(s).apply {
                isChecked = i == selected
                setOnClickListener {
                    for (k in 0 until row.childCount) (row.getChildAt(k) as Chip).isChecked = k == i
                    onSelect(i)
                }
            }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginEnd = dp(8) })
        }
        val scroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            addView(row)
        }
        return scroll to row
    }

    private fun button(text: String, primary: Boolean = false, onClick: () -> Unit) =
        MaterialButton(this, null, if (primary) com.google.android.material.R.attr.materialButtonStyle else com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            this.text = text
            setOnClickListener { onClick() }
        }

    // ---------------------------------------------------------------- 工具

    private abstract inner class Tool(val title: String) {
        /** 进入工具时的预览图，所有实时预览都基于它 */
        val basePreview: Bitmap = preview!!
        val basePx: IntArray by lazy { ImageIO.pixels(basePreview) }
        val pw get() = basePreview.width
        val ph get() = basePreview.height
        open val applyText = "正在处理…"
        abstract fun panel(): View
        open fun enter() {}
        open fun exit() {}
        open fun onResume() {}

        /** 按原图尺寸算出结果；null 表示没有改动 */
        abstract suspend fun result(): Bitmap?
    }

    private inner class EnhanceTool : Tool("智能增强") {
        @Volatile private var enhanced: IntArray? = null
        private var strength = 0.7f

        override fun panel() = column(
            hint("自动校正曝光、白平衡、对比度和清晰度，在手机上完成"),
            sliderRow("强度", 0f, 100f, 70f) { strength = it / 100f; refresh() }.first,
        )

        override fun enter() {
            val base = basePx
            render {
                val e = Enhance.auto(base, pw, ph)
                enhanced = e
                Pixels.mix(base, e, strength)
            }
        }

        private fun refresh() {
            val e = enhanced ?: return
            val t = strength
            render { Pixels.mix(basePx, e, t) }
        }

        override suspend fun result(): Bitmap? = withContext(Dispatchers.Default) {
            if (strength <= 0f) return@withContext null
            val b = current!!
            val px = ImageIO.pixels(b)
            ImageIO.toBitmap(Pixels.mix(px, Enhance.auto(px, b.width, b.height), strength), b.width, b.height)
        }
    }

    private inner class AdjustTool : Tool("调整") {
        private inner class Param(val name: String, val positiveOnly: Boolean, val get: (Adjust) -> Float, val set: (Adjust, Float) -> Adjust)

        private val params = listOf(
            Param("亮度", false, { it.brightness }) { a, v -> a.copy(brightness = v) },
            Param("对比度", false, { it.contrast }) { a, v -> a.copy(contrast = v) },
            Param("高光", false, { it.highlights }) { a, v -> a.copy(highlights = v) },
            Param("阴影", false, { it.shadows }) { a, v -> a.copy(shadows = v) },
            Param("饱和度", false, { it.saturation }) { a, v -> a.copy(saturation = v) },
            Param("色温", false, { it.warmth }) { a, v -> a.copy(warmth = v) },
            Param("色调", false, { it.tint }) { a, v -> a.copy(tint = v) },
            Param("褪色", true, { it.fade }) { a, v -> a.copy(fade = v) },
            Param("锐化", true, { it.sharpen }) { a, v -> a.copy(sharpen = v) },
            Param("暗角", false, { -it.vignette }) { a, v -> a.copy(vignette = -v) },
        )
        private var adj = Adjust()
        private var selected = 0
        private lateinit var slider: Slider
        private lateinit var chips: LinearLayout

        private fun chipText(p: Param): String {
            val v = (p.get(adj) * 100).roundToInt()
            return if (v == 0) p.name else p.name + " " + (if (v > 0) "+$v" else "$v")
        }

        override fun panel(): View {
            val (row, group) = chipRow(params.map { it.name }, 0) { select(it) }
            chips = group
            val (sRow, s) = sliderRow("", -100f, 100f, 0f, { v -> (if (v > 0) "+" else "") + v.roundToInt() }) { v ->
                val p = params[selected]
                adj = p.set(adj, v / 100f)
                (chips.getChildAt(selected) as Chip).text = chipText(p)
                val a = adj
                render { a.apply(basePx, pw, ph) }
            }
            slider = s
            return column(row, sRow)
        }

        private fun select(i: Int) {
            selected = i
            val p = params[i]
            val v = p.get(adj) * 100
            slider.valueFrom = if (p.positiveOnly) 0f else -100f
            slider.value = v.coerceIn(slider.valueFrom, 100f)
        }

        override suspend fun result(): Bitmap? = withContext(Dispatchers.Default) {
            if (adj.isIdentity) return@withContext null
            val b = current!!
            ImageIO.toBitmap(adj.apply(ImageIO.pixels(b), b.width, b.height), b.width, b.height)
        }
    }

    private inner class FilterTool : Tool("滤镜") {
        private var selected = 0
        private var intensity = 1f
        private lateinit var strengthRow: View
        private val items = ArrayList<View>()

        override fun panel(): View {
            val (row, _) = sliderRow("强度", 0f, 100f, 100f) {
                intensity = it / 100f
                refresh()
            }
            strengthRow = row
            row.visibility = View.INVISIBLE
            val list = LinearLayout(this@EditorActivity)
            val thumbSrc = ImageIO.fit(basePreview, 180)
            val tpx = ImageIO.pixels(thumbSrc)
            Filters.ALL.forEachIndexed { i, f ->
                val img = ImageView(this@EditorActivity).apply {
                    scaleType = ImageView.ScaleType.CENTER_CROP
                    setImageBitmap(thumbSrc)
                    clipToOutline = true
                    outlineProvider = object : ViewOutlineProvider() {
                        override fun getOutline(view: View, outline: Outline) = outline.setRoundRect(0, 0, view.width, view.height, dp(10).toFloat())
                    }
                }
                val label = TextView(this@EditorActivity).apply {
                    text = f.name
                    textSize = 12f
                    gravity = Gravity.CENTER
                    setPadding(0, dp(4), 0, 0)
                }
                val item = LinearLayout(this@EditorActivity).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(dp(4), dp(4), dp(4), dp(4))
                    addView(img, LinearLayout.LayoutParams(dp(64), dp(64)))
                    addView(label, LinearLayout.LayoutParams(dp(64), ViewGroup.LayoutParams.WRAP_CONTENT))
                    setOnClickListener { select(i) }
                }
                items += item
                list.addView(item)
                if (i > 0) lifecycleScope.launch {
                    val b = withContext(Dispatchers.Default) { ImageIO.toBitmap(f.adjust.apply(tpx, thumbSrc.width, thumbSrc.height), thumbSrc.width, thumbSrc.height) }
                    img.setImageBitmap(b)
                }
            }
            highlight()
            val scroll = HorizontalScrollView(this@EditorActivity).apply {
                isHorizontalScrollBarEnabled = false
                addView(list)
            }
            return column(row, scroll)
        }

        private fun highlight() {
            items.forEachIndexed { i, v ->
                val label = (v as LinearLayout).getChildAt(1) as TextView
                label.setTextColor(ContextCompat.getColor(this@EditorActivity, if (i == selected) R.color.accent else R.color.text2))
                v.getChildAt(0).alpha = if (i == selected) 1f else 0.85f
            }
        }

        private fun select(i: Int) {
            selected = i
            strengthRow.visibility = if (i == 0) View.INVISIBLE else View.VISIBLE
            highlight()
            refresh()
        }

        private fun refresh() {
            val a = Filters.ALL[selected].adjust.scaled(intensity)
            render { a.apply(basePx, pw, ph) }
        }

        override suspend fun result(): Bitmap? = withContext(Dispatchers.Default) {
            if (selected == 0 || intensity <= 0f) return@withContext null
            val b = current!!
            val a = Filters.ALL[selected].adjust.scaled(intensity)
            ImageIO.toBitmap(a.apply(ImageIO.pixels(b), b.width, b.height), b.width, b.height)
        }
    }

    private inner class CropTool : Tool("裁剪旋转") {
        private val ratios = listOf("自由" to 0f, "原比例" to -1f, "1:1" to 1f, "4:3" to 4 / 3f, "3:4" to 3 / 4f, "16:9" to 16 / 9f, "9:16" to 9 / 16f)
        private var ratio = 0
        private val transform = Matrix()
        private var shown: Bitmap = basePreview

        override fun panel(): View {
            val (row, _) = chipRow(ratios.map { it.first }, 0) {
                ratio = it
                photo.setCropAspect(aspect())
            }
            val buttons = LinearLayout(this@EditorActivity).apply {
                gravity = Gravity.CENTER
                setPadding(0, dp(6), 0, 0)
                addView(button("旋转 90°") { transform.postRotate(90f); reshow() })
                addView(View(context), LinearLayout.LayoutParams(dp(12), 1))
                addView(button("水平翻转") { transform.postScale(-1f, 1f); reshow() })
                addView(View(context), LinearLayout.LayoutParams(dp(12), 1))
                addView(button("还原") { transform.reset(); ratio = 0; reshow() })
            }
            return column(row, buttons)
        }

        private fun aspect(): Float {
            val r = ratios[ratio].second
            return if (r < 0) shown.width.toFloat() / shown.height else r
        }

        override fun enter() {
            photo.mode = PhotoView.Mode.CROP
            reshow()
        }

        private fun reshow() {
            shown = if (transform.isIdentity) basePreview else Bitmap.createBitmap(basePreview, 0, 0, pw, ph, transform, true)
            photo.setImage(shown)
            photo.setCropAspect(aspect())
        }

        /** 只转了 0° / 360° 且没有翻转（或翻了两次）时视为没有变换 */
        private fun noTransform(): Boolean {
            val v = floatArrayOf(1f, 0f, 0f, 1f)
            transform.mapVectors(v)
            return abs(v[0] - 1) < 1e-3 && abs(v[1]) < 1e-3 && abs(v[2]) < 1e-3 && abs(v[3] - 1) < 1e-3
        }

        override suspend fun result(): Bitmap? = withContext(Dispatchers.Default) {
            val c = photo.crop
            val full = c.left <= 0.001f && c.top <= 0.001f && c.right >= 0.999f && c.bottom >= 0.999f
            if (full && noTransform()) return@withContext null
            val b = current!!
            val t = if (noTransform()) b else Bitmap.createBitmap(b, 0, 0, b.width, b.height, transform, true)
            if (full) return@withContext t
            val l = (c.left * t.width).roundToInt().coerceIn(0, t.width - 1)
            val top = (c.top * t.height).roundToInt().coerceIn(0, t.height - 1)
            val r = (c.right * t.width).roundToInt().coerceIn(l + 1, t.width)
            val bottom = (c.bottom * t.height).roundToInt().coerceIn(top + 1, t.height)
            ImageIO.ensureArgb(Bitmap.createBitmap(t, l, top, r - l, bottom - top))
        }
    }

    private inner class CutoutTool : Tool("人像抠图") {
        private var mask: Segment.Mask? = null
        private var effect = Segment.Effect.BLUR
        private var blur = 0.5f
        private lateinit var blurRow: View

        override fun panel(): View {
            val effects = Segment.Effect.entries
            val (row, _) = chipRow(effects.map { it.label }, 0) {
                effect = effects[it]
                blurRow.visibility = if (effect == Segment.Effect.BLUR) View.VISIBLE else View.INVISIBLE
                refresh()
            }
            val (bRow, _) = sliderRow("虚化", 0f, 100f, 50f) {
                blur = it / 100f
                refresh()
            }
            blurRow = bRow
            return column(hint("识别照片里的人（本机运行），处理背景"), row, bRow)
        }

        override fun enter() {
            busy("正在识别人像…") {
                val m = Segment.person(basePreview)
                if (Cutout.coverage(m.data) < 0.005f) {
                    closeTool()
                    alert("没有识别到人", "抠图目前只支持有人物的照片。去掉别的东西可以用「消除」，换背景也可以在「AI 修图」里直接描述。")
                    return@busy
                }
                mask = m
                refresh()
            }
        }

        private fun refresh() {
            val m = mask ?: return
            val e = effect
            val s = blur
            render { ImageIO.pixels(Segment.render(basePreview, m, e, s)) }
        }

        override suspend fun result(): Bitmap? {
            val m = mask ?: return null
            return withContext(Dispatchers.Default) { Segment.render(current!!, m, effect, blur) }
        }
    }

    private inner class EraseTool : Tool("消除") {
        /** 这次进来后已经消除过的整图 */
        private var pending: Bitmap? = null
        private lateinit var undoStroke: View
        private lateinit var quick: View
        private lateinit var ai: View

        override fun panel(): View {
            val (sizeRow, _) = sliderRow("笔刷", 8f, 60f, 22f) { photo.brushRadius = it * resources.displayMetrics.density }
            undoStroke = button("撤销一笔") { photo.undoStroke() }
            quick = button("快速消除") { quick() }
            ai = button("AI 消除", primary = true) { ai() }
            val buttons = LinearLayout(this@EditorActivity).apply {
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, dp(4), 0, 0)
                addView(undoStroke)
                addView(View(context), LinearLayout.LayoutParams(0, 1, 1f))
                addView(quick)
                addView(View(context), LinearLayout.LayoutParams(dp(8), 1))
                addView(ai)
            }
            return column(hint("涂抹要去掉的东西，双指缩放移动。快速消除在手机上完成，适合小瑕疵；AI 消除效果更好，需要联网"), sizeRow, buttons)
        }

        override fun enter() {
            photo.mode = PhotoView.Mode.BRUSH
            photo.brushRadius = 22 * resources.displayMetrics.density
            photo.onMaskChanged = { updateEnabled() }
            photo.resetMask()
        }

        override fun exit() {
            photo.onMaskChanged = null
        }

        private fun updateEnabled() {
            val has = photo.hasMask
            for (v in listOf(undoStroke, quick, ai)) {
                v.isEnabled = has
            }
        }

        private fun quick() {
            val mb = photo.maskBitmap() ?: return
            busy("正在消除…") {
                val base = pending ?: current!!
                val r = withContext(Dispatchers.Default) {
                    val m = Cloud.alphaOf(ImageIO.scaled(mb, base.width, base.height))
                    ImageIO.toBitmap(Inpaint.fill(ImageIO.pixels(base), base.width, base.height, m), base.width, base.height)
                }
                show(r)
            }
        }

        private fun ai() {
            val mb = photo.maskBitmap() ?: return
            val cfg = Store.cloud() ?: return askConfigure()
            busy("AI 正在消除…\n通常需要 10～60 秒", cancellable = true) {
                val r = Cloud.erase(cfg, pending ?: current!!, mb)
                show(r)
            }
        }

        private suspend fun show(r: Bitmap) {
            pending = r
            val p = withContext(Dispatchers.Default) { ImageIO.fit(r, PREVIEW_SIDE) }
            photo.setImage(p)
            photo.resetMask()
        }

        override suspend fun result(): Bitmap? = pending
    }

    private fun askConfigure() {
        MaterialAlertDialogBuilder(this)
            .setTitle("需要设置 AI 服务")
            .setMessage("AI 功能使用你自己的 Google Gemini 或 OpenAI（及兼容接口）的 API Key，请先在设置里填写。")
            .setPositiveButton("去设置") { _, _ -> startActivity(Intent(this, SettingsActivity::class.java)) }
            .setNegativeButton("取消", null)
            .show()
    }

    private inner class AiTool : Tool("AI 修图") {
        private var result: Bitmap? = null
        private var input: TextInputEditText? = null
        private val presets = listOf(
            "画质增强" to "提高清晰度和细节，去除噪点和模糊，色彩自然真实",
            "老照片修复" to "修复这张老照片：去除划痕、折痕、污渍和噪点，提高清晰度，并自然地上色",
            "去掉路人" to "去掉背景里所有的路人，只保留主体人物，背景自然补全",
            "换成蓝天" to "把天空换成晴朗的蓝天白云，整体光线和色调相应协调",
            "晚霞天空" to "把天空换成绚丽的晚霞，整体变成黄昏的暖色光线",
            "证件照白底" to "把背景换成纯白色，人物不变，光线均匀，适合做证件照",
            "动漫风格" to "把整张照片转成日系动漫插画风格，人物和构图保持不变",
            "去除反光" to "去掉眼镜和画面里的反光，其它保持不变",
        )

        override fun panel(): View {
            val cfg = Store.cloud()
            if (cfg == null) {
                return column(
                    hint("「AI 修图」用一句话描述想怎么改照片，由云端图像模型完成。需要先在设置里填写 Gemini 或 OpenAI 的 API Key。"),
                    button("去设置", primary = true) { startActivity(Intent(this@EditorActivity, SettingsActivity::class.java)) },
                )
            }
            val group = LinearLayout(this@EditorActivity).apply { gravity = Gravity.CENTER_VERTICAL }
            val recent = Store.recentPrompts().map { (if (it.length > 10) it.take(10) + "…" else it) to it }
            for ((label, prompt) in recent.map { "↺ " + it.first to it.second } + presets) {
                group.addView(Chip(this@EditorActivity).apply {
                    text = label
                    setEnsureMinTouchTargetSize(false)
                    setOnClickListener {
                        input?.setText(prompt)
                        input?.setSelection(prompt.length)
                    }
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginEnd = dp(8) })
            }
            val scroll = HorizontalScrollView(this@EditorActivity).apply {
                isHorizontalScrollBarEnabled = false
                addView(group)
            }
            val layout = TextInputLayout(this@EditorActivity, null, com.google.android.material.R.attr.textInputOutlinedStyle).apply {
                hint = "想怎么改？例如：把天空换成晚霞"
            }
            val edit = TextInputEditText(layout.context).apply {
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                maxLines = 3
                textSize = 14f
            }
            input = edit
            layout.addView(edit)
            val row = LinearLayout(this@EditorActivity).apply {
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, dp(6), 0, 0)
                addView(layout, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                addView(View(context), LinearLayout.LayoutParams(dp(8), 1))
                addView(button("生成", primary = true) { generate() })
            }
            return column(
                scroll,
                row,
                hint("照片会发送到 ${Store.providerName(cfg.provider)}（${cfg.model}），可在设置里更改").apply { setPadding(0, dp(4), 0, 0) },
            )
        }

        /** 从设置页回来：可能刚填好 Key，重建面板 */
        override fun onResume() {
            if (input == null && Store.cloud() != null) {
                panel.removeAllViews()
                panel.addView(panel())
            }
        }

        private fun generate() {
            val prompt = input?.text?.toString()?.trim().orEmpty()
            if (prompt.isEmpty()) {
                toast("先写一句想怎么改")
                return
            }
            val cfg = Store.cloud() ?: return askConfigure()
            Store.addRecentPrompt(prompt)
            hideKeyboard()
            busy("AI 正在修图…\n通常需要 10～60 秒", cancellable = true) {
                val r = Cloud.edit(cfg, current!!, prompt)
                result = r
                photo.setImage(withContext(Dispatchers.Default) { ImageIO.fit(r, PREVIEW_SIDE) })
                flashInfo("结果 ${r.width} × ${r.height}，满意就点 ✓", 4000)
            }
        }

        override suspend fun result(): Bitmap? = result
    }
}
