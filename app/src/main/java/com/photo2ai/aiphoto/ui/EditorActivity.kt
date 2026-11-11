package com.photo2ai.aiphoto.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Outline
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
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
import androidx.activity.result.PickVisualMediaRequest
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
import com.photo2ai.aiphoto.core.Accel
import com.photo2ai.aiphoto.core.Adjust
import com.photo2ai.aiphoto.core.Beauty
import com.photo2ai.aiphoto.core.Cloud
import com.photo2ai.aiphoto.core.CloudProtocol
import com.photo2ai.aiphoto.core.Cutout
import com.photo2ai.aiphoto.core.Enhance
import com.photo2ai.aiphoto.core.FaceMesh
import com.photo2ai.aiphoto.core.Garment
import com.photo2ai.aiphoto.core.Filters
import com.photo2ai.aiphoto.core.History
import com.photo2ai.aiphoto.core.ImageIO
import com.photo2ai.aiphoto.core.Inpaint
import com.photo2ai.aiphoto.core.LocalAi
import com.photo2ai.aiphoto.core.LocalModels
import com.photo2ai.aiphoto.core.Matting
import com.photo2ai.aiphoto.core.NeuralSwap
import com.photo2ai.aiphoto.core.Pixels
import com.photo2ai.aiphoto.core.Segment
import com.photo2ai.aiphoto.core.Store
import com.photo2ai.aiphoto.core.TryOn
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

        /** 抠图工具一打开就选中的效果（Segment.Effect 的名字），首页「证件照」用 */
        const val EXTRA_EFFECT = "effect"
        const val TOOL_ENHANCE = "enhance"
        const val TOOL_BEAUTY = "beauty"

        /** 高清换脸用的模型，顺序对应 [FaceMesh.neuralSwap] 的参数 */
        private val SWAP_MODELS = listOf(LocalModels.ARCFACE, LocalModels.HYPERSWAP, LocalModels.GFPGAN)
        /** 本地换装要用的模型，顺序和 TryOn.Models 的参数一致 */
        private val TRYON_MODELS = listOf(LocalModels.MODNET, LocalModels.MIGAN, LocalModels.PARSING, LocalModels.DEPTH, LocalModels.HARMONIZER)
        const val TOOL_AI = "ai"
        const val TOOL_ERASE = "erase"
        const val TOOL_CUTOUT = "cutout"
        const val TOOL_ADJUST = "adjust"
        const val TOOL_FILTER = "filter"
        const val TOOL_CROP = "crop"
        const val TOOL_FACE = "face"
        const val TOOL_UPSCALE = "upscale"

        /** 不是编辑页里的工具：把照片交给系统相册的修图，修完导入回来 */
        const val TOOL_SYSTEM = "system"
        const val TOOL_OUTFIT = "outfit"

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
    private fun busy(text: String, cancellable: Boolean = false, block: suspend () -> Unit): Job {
        busyJob?.takeIf { it.isActive }?.let { return it }
        showBusy(text, cancellable)
        return lifecycleScope.launch {
            try {
                block()
            } catch (e: CancellationException) {
                throw e
            } catch (e: OutOfMemoryError) {
                alert("内存不足", "这张图太大，处理时内存不够。可以先裁小一些再试。")
            } catch (e: CloudProtocol.Failure) {
                alert("AI 处理失败", e.message ?: "")
            } catch (e: LocalAi.BrokenModel) {
                alert("模型出错", e.message ?: "")
            } catch (e: Exception) {
                android.util.Log.w("AIPhoto", "$text failed", e)
                alert("出错了", e.message ?: e.javaClass.simpleName)
            } finally {
                hideBusy()
            }
        }.also { busyJob = it }
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
        if (modelDialog != null) {
            modelDialog = null
            ModelDownloads.detached(this)
        }
        if (isFinishing) history.clear()
    }

    // ---------------------------------------------------------------- 工具栏

    private class ToolDef(val id: String, val icon: Int, val label: String)

    private val toolDefs = listOf(
        ToolDef(TOOL_ENHANCE, R.drawable.ic_magic, "智能增强"),
        ToolDef(TOOL_BEAUTY, R.drawable.ic_beauty, "美颜"),
        ToolDef(TOOL_UPSCALE, R.drawable.ic_hd, "变清晰"),
        ToolDef(TOOL_AI, R.drawable.ic_sparkle, "AI 修图"),
        ToolDef(TOOL_FACE, R.drawable.ic_face, "换脸"),
        ToolDef(TOOL_OUTFIT, R.drawable.ic_shirt, "换装"),
        ToolDef(TOOL_ERASE, R.drawable.ic_eraser, "消除笔"),
        ToolDef(TOOL_CUTOUT, R.drawable.ic_person, "抠图"),
        ToolDef(TOOL_ADJUST, R.drawable.ic_tune, "调整"),
        ToolDef(TOOL_FILTER, R.drawable.ic_filter, "滤镜"),
        ToolDef(TOOL_CROP, R.drawable.ic_crop, "裁剪"),
        ToolDef(TOOL_SYSTEM, R.drawable.ic_gallery, "相册修图"),
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
                    if (d.id == TOOL_AI || d.id == TOOL_FACE || d.id == TOOL_OUTFIT) setColorFilter(ContextCompat.getColor(context, R.color.accent))
                }, LinearLayout.LayoutParams(dp(26), dp(26)))
                addView(TextView(context).apply {
                    text = d.label
                    textSize = 12f
                    setTextColor(ContextCompat.getColor(context, R.color.ed_text))
                    setPadding(0, dp(6), 0, 0)
                })
            })
        }
    }

    private fun openTool(id: String) {
        if (tool != null || current == null || busyJob?.isActive == true) return
        if (id == TOOL_SYSTEM) return systemEdit()
        val t = when (id) {
            TOOL_ENHANCE -> EnhanceTool()
            TOOL_BEAUTY -> BeautyTool()
            TOOL_AI -> AiTool()
            TOOL_ERASE -> EraseTool()
            TOOL_CUTOUT -> CutoutTool()
            TOOL_ADJUST -> AdjustTool()
            TOOL_FILTER -> FilterTool()
            TOOL_CROP -> CropTool()
            TOOL_FACE -> FaceTool()
            TOOL_UPSCALE -> UpscaleTool()
            TOOL_OUTFIT -> OutfitTool()
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
        setTextColor(ContextCompat.getColor(context, R.color.ed_text2))
        setPadding(0, 0, 0, dp(4))
    }

    /** 名称 + 数值 + 滑杆 */
    private fun sliderRow(label: String, from: Float, to: Float, value: Float, format: (Float) -> String = { it.roundToInt().toString() }, onChange: (Float) -> Unit): Pair<View, Slider> {
        val valueText = TextView(this).apply {
            text = format(value)
            textSize = 13f
            setTextColor(ContextCompat.getColor(context, R.color.ed_text))
            minWidth = dp(36)
            gravity = Gravity.END
        }
        val slider = Slider(this).apply {
            valueFrom = from
            valueTo = to
            this.value = value
            stepSize = 1f
            isTickVisible = false
            labelBehavior = LabelFormatter.LABEL_FLOATING
            setLabelFormatter { format(it) }
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
                setTextColor(ContextCompat.getColor(context, R.color.ed_text2))
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
        val accent = ContextCompat.getColor(context, R.color.ed_accent_soft)
        val normal = ContextCompat.getColor(context, R.color.ed_surface2)
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

    private inner class BeautyTool : Tool("美颜") {
        private inner class Param(val name: String, val get: (Beauty.Params) -> Float, val set: (Beauty.Params, Float) -> Beauty.Params)

        private val params = listOf(
            Param("磨皮", { it.smooth }) { a, v -> a.copy(smooth = v) },
            Param("美白", { it.whiten }) { a, v -> a.copy(whiten = v) },
            Param("瘦脸", { it.slim }) { a, v -> a.copy(slim = v) },
            Param("大眼", { it.eyes }) { a, v -> a.copy(eyes = v) },
        )

        /** 一打开就有磨皮、美白，瘦脸、大眼让用户自己加 */
        private var values = Beauty.Params(smooth = 0.5f, whiten = 0.3f)
        private var selected = 0
        private var faces: List<FaceMesh.Face> = emptyList()
        private var prepared: List<Beauty.Prepared> = emptyList()
        private lateinit var slider: Slider
        private lateinit var chips: LinearLayout
        override val applyText = "正在美颜…"

        private fun chipText(p: Param): String {
            val v = (p.get(values) * 100).roundToInt()
            return if (v == 0) p.name else "${p.name} $v"
        }

        override fun panel(): View {
            val (row, group) = chipRow(params.map { chipText(it) }, 0) { select(it) }
            chips = group
            val (sRow, s) = sliderRow("", 0f, 100f, params[0].get(values) * 100) { v ->
                val p = params[selected]
                values = p.set(values, v / 100f)
                (chips.getChildAt(selected) as Chip).text = chipText(p)
                refresh()
            }
            slider = s
            return column(hint("磨皮（也会淡化斑点、痘印）和美白只作用在脸部皮肤上，在手机上完成"), row, sRow)
        }

        private fun select(i: Int) {
            selected = i
            slider.value = params[i].get(values) * 100
        }

        override fun enter() {
            busy("正在识别人脸…") {
                val found = FaceMesh.detectAll(basePreview)
                val base = basePx
                val ok = withContext(Dispatchers.Default) {
                    found.mapNotNull { f -> Beauty.prepare(base, pw, ph, f.beautyGeom())?.let { f to it } }
                }
                if (ok.isEmpty()) {
                    closeTool()
                    alert("没有找到人脸", "美颜需要照片里有清楚的人脸，脸太小也找不到。")
                    return@busy
                }
                faces = ok.map { it.first }
                prepared = ok.map { it.second }
                if (ok.size > 1) flashInfo("找到 ${ok.size} 张脸，一起美颜")
                refresh()
            }
        }

        private fun refresh() {
            val list = prepared
            if (list.isEmpty()) return
            val v = values
            val base = basePx
            render { Beauty.apply(base, pw, list, v) }
        }

        /** 原图上重新准备一遍：人脸网格按预览图到原图的比例放大，半径、力度都跟着脸宽走，效果和预览一致 */
        override suspend fun result(): Bitmap? = withContext(Dispatchers.Default) {
            if (values.isIdentity || faces.isEmpty()) return@withContext null
            val b = current!!
            val k = b.width.toFloat() / pw
            val px = ImageIO.pixels(b)
            val prep = faces.mapNotNull { Beauty.prepare(px, b.width, b.height, it.beautyGeom().scaled(k)) }
            ImageIO.toBitmap(Beauty.apply(px, b.width, prep, values), b.width, b.height)
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
                label.setTextColor(ContextCompat.getColor(this@EditorActivity, if (i == selected) R.color.accent else R.color.ed_text2))
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

    private inner class UpscaleTool : Tool("变清晰") {
        private var factor = 2
        private var result: Bitmap? = null
        private lateinit var sizeText: TextView

        override fun panel(): View {
            val (row, _) = chipRow(listOf("放大 2 倍", "放大 4 倍"), 0) {
                factor = if (it == 0) 2 else 4
                updateSize()
            }
            sizeText = hint("")
            updateSize()
            val go = button("开始", primary = true) { go() }
            return column(
                hint("本地 AI 模型放大照片，同时去模糊、去噪点，适合小图、截图、老照片；照片不会上传"),
                row,
                LinearLayout(this@EditorActivity).apply {
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(0, dp(6), 0, 0)
                    addView(sizeText, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                    addView(go)
                },
            )
        }

        /** 送进模型的尺寸：2 倍时先缩小一半再放大 4 倍；输出不超过可编辑的最大像素，输入不超过 100 万像素（手机上要算几十秒） */
        private fun inputSize(): Pair<Int, Int> {
            val b = current!!
            var w = b.width * factor / 4.0
            var h = b.height * factor / 4.0
            val maxIn = minOf(1_000_000.0, ImageIO.maxPixels() / 16.0)
            if (w * h > maxIn) {
                val k = Math.sqrt(maxIn / (w * h))
                w *= k
                h *= k
            }
            return maxOf(1, w.toInt()) to maxOf(1, h.toInt())
        }

        private fun updateSize() {
            val b = current!!
            val (w, h) = inputSize()
            sizeText.text = "${b.width} × ${b.height} → ${w * 4} × ${h * 4}" + if (w * 4 < b.width * factor) "（已按内存限制）" else ""
        }

        private fun go() {
            if (!LocalModels.supported()) return alert("不支持", "这台手机是 32 位系统，用不了本地 AI 模型。")
            withModel(LocalModels.ESRGAN) { model ->
                busy("正在变清晰…", cancellable = true) {
                    val (w, h) = inputSize()
                    val r = withContext(Dispatchers.Default) {
                        val src = ImageIO.scaled(current!!, w, h)
                        val out = LocalAi.upscale4x(model, ImageIO.pixels(src), w, h) { p ->
                            runOnUiThread { busyText.text = "正在变清晰… ${(p * 100).toInt()}%" }
                        }
                        ImageIO.toBitmap(out, w * 4, h * 4)
                    }
                    result = r
                    photo.setImage(withContext(Dispatchers.Default) { ImageIO.fit(r, PREVIEW_SIDE) })
                    flashInfo("已放大到 ${r.width} × ${r.height}，满意就点 ✓（按住右下角对比）", 4000)
                }
            }
        }

        override suspend fun result(): Bitmap? = result
    }

    /** 用一个本地模型前确认已下载，见 [withModels] */
    private fun withModel(m: LocalModels.Model, then: (File) -> Unit) = withModels(listOf(m), null) { then(it[0]) }

    /**
     * 用本地模型前确认都已下载：缺的一起提示总大小（[note] 是额外说明，比如许可），确认后交给 [ModelDownloads] 下载。
     * 进度弹窗不收起就一直等，下完接着执行 [then]（参数顺序和 [models] 一致）；收起了（后台下载）就不再自动执行，
     * 下完发通知，用户再点一次即可。
     * [skip] 不为 null 时下载框多一个按钮（文字是 skip 的第一项），不下载、改用不需要模型的做法；
     * [onCancel] 是不下载也不跳过（点取消、返回、或在等别的模型）时要做的事
     */
    private fun withModels(
        models: List<LocalModels.Model>, note: String?, skip: Pair<String, () -> Unit>? = null,
        onCancel: (() -> Unit)? = null, then: (List<File>) -> Unit,
    ) {
        // 还有别的处理在跑时 busy() 不会开新任务：先拦下，免得「下载」点了没反应
        if (busyJob?.isActive == true) return toast("请等当前的处理完成").also { onCancel?.invoke() }
        if (ModelDownloads.running) {
            // 正在下（可能是刚才收到后台的）：要的就是这一批就把进度拿出来接着等，否则先等那一批下完
            if (models.any { it in ModelDownloads.models }) return showModelProgress { measureThen(models, then) }
            return toast("正在下载其它模型，下完再用这个功能").also { onCancel?.invoke() }
        }
        val missing = models.filter { LocalModels.ready(this, it) == null }
        if (missing.isEmpty()) return measureThen(models, then)
        val mb = "%.0f".format(missing.sumOf { it.size } / 1048576.0)
        val kept = missing.sumOf { LocalModels.partialBytes(this, it) }
        val names = missing.joinToString("、") { it.title }
        MaterialAlertDialogBuilder(this)
            .setTitle("下载模型")
            .setMessage(
                "第一次使用需要下载$names，共约 $mb MB，只下载一次，之后离线可用。建议在 Wi-Fi 下下载，可以放到后台下。" +
                    (if (kept > 0) "\n已下载 %.1f MB，会接着下。".format(kept / 1048576.0) else "") +
                    (note?.let { "\n\n$it" } ?: ""),
            )
            .setPositiveButton(if (kept > 0) "继续下载" else "下载") { _, _ ->
                if (busyJob?.isActive == true) return@setPositiveButton toast("请等当前的处理完成")
                if (Build.VERSION.SDK_INT >= 33 &&
                    ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
                ) {
                    requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
                }
                ModelDownloads.start(this, missing)
                showModelProgress { measureThen(models, then) }
            }
            .setNegativeButton("取消") { _, _ -> onCancel?.invoke() }
            .setOnCancelListener { onCancel?.invoke() }
            .apply { skip?.let { (label, run) -> setNeutralButton(label) { _, _ -> run() } } }
            .show()
    }

    private var modelDialog: androidx.appcompat.app.AlertDialog? = null

    /** 模型下载的进度弹窗：「后台下载」收起（之后由通知显示进度），「取消」停止（已下的部分留着续传） */
    private fun showModelProgress(onDone: () -> Unit) {
        modelDialog?.dismiss()
        val bar = com.google.android.material.progressindicator.LinearProgressIndicator(this).apply { isIndeterminate = true; max = 100 }
        val label = TextView(this).apply { setPadding(0, dp(10), 0, 0) }
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24), dp(16), dp(24), 0)
            addView(bar)
            addView(label)
        }
        var waiting = true
        val dialog = MaterialAlertDialogBuilder(this)
            .setTitle("正在下载模型")
            .setView(box)
            .setCancelable(false)
            .setPositiveButton("后台下载") { _, _ -> toast("在后台下载，好了会通知你") }
            .setNegativeButton("取消") { _, _ -> ModelDownloads.cancel() }
            .setOnDismissListener {
                if (modelDialog === it) modelDialog = null
                // 收起或取消：不再等这一批的结果
                if (waiting) {
                    waiting = false
                    ModelDownloads.detached(this)
                }
            }
            .show()
        modelDialog = dialog
        val show = { st: ModelDownloads.State ->
            if (st.finished) {
                val wasWaiting = waiting
                waiting = false
                ModelDownloads.listener = null
                dialog.dismiss()
                when {
                    st.error == "已取消" -> Unit
                    st.error != null -> alert("模型下载中断", st.error)
                    wasWaiting && !isFinishing -> onDone()
                }
            } else {
                dialog.setTitle("正在下载" + st.label)
                if (st.percent >= 0) {
                    bar.isIndeterminate = false
                    bar.setProgressCompat(st.percent, true)
                }
                label.text = "%.1f / %.1f MB".format(st.done / 1048576.0, maxOf(st.total, st.done) / 1048576.0)
            }
        }
        ModelDownloads.listener = show
        ModelDownloads.state?.let(show)
    }

    /**
     * 模型都就绪后：这台手机上还没测过的，先测一次用 CPU 还是 AI 芯片更快（每个模型只测一次，几秒到几十秒），再执行 [then]。
     * 测速出错不影响使用，按 CPU 来。
     */
    private fun measureThen(models: List<LocalModels.Model>, then: (List<File>) -> Unit) {
        val files = models.map { LocalModels.ready(this, it)!! }
        val todo = models.filter { !Accel.measured(it) }
        if (todo.isEmpty() || !LocalModels.supported()) return then(files)
        var done = false
        busy("正在测试这台手机的 AI 加速…\n每个模型只测一次") {
            withContext(Dispatchers.Default) {
                todo.forEachIndexed { k, m ->
                    runOnUiThread { busyText.text = "正在测试这台手机的 AI 加速（${k + 1}/${todo.size}）…\n每个模型只测一次" }
                    try {
                        Accel.benchmark(m, files[models.indexOf(m)])
                    } catch (e: Exception) {
                        android.util.Log.w("AIPhoto", "benchmark ${m.file} failed", e)
                        Store.setAccel(m.sha256, Accel.Backend.CPU.name, m.title + "：CPU（测速出错）")
                    }
                }
            }
            LocalAi.releaseAll()
            done = true
        }.invokeOnCompletion { e -> if (e == null && done) runOnUiThread { then(files) } }
    }

    private inner class CutoutTool : Tool("人像抠图") {
        private var mask: Segment.Mask? = null

        /** 预览图上估好的前景颜色（换底时去掉边缘的原背景色），拖滑杆时不重算 */
        private var previewFg: IntArray? = null
        private var effect = intent.getStringExtra(EXTRA_EFFECT)
            ?.let { name -> Segment.Effect.entries.firstOrNull { it.name == name } }
            ?.also { intent.removeExtra(EXTRA_EFFECT) } // 只在第一次打开时生效
            ?: Segment.Effect.BLUR
        private var blur = 0.5f
        private lateinit var blurRow: View

        override fun panel(): View {
            val effects = Segment.Effect.entries
            val (row, _) = chipRow(effects.map { it.label }, effects.indexOf(effect)) {
                effect = effects[it]
                blurRow.visibility = if (effect == Segment.Effect.BLUR) View.VISIBLE else View.INVISIBLE
                refresh()
            }
            val (bRow, _) = sliderRow("虚化", 0f, 100f, 50f) {
                blur = it / 100f
                refresh()
            }
            blurRow = bRow
            if (effect != Segment.Effect.BLUR) bRow.visibility = View.INVISIBLE
            return column(hint("识别照片里的人（本机运行），处理背景；下载精细抠图模型后头发丝也能抠出来"), row, bRow)
        }

        override fun enter() {
            // 精细抠图要下载约 12 MB 的 MODNet；不想下载（或 32 位手机用不了）就用 ML Kit 普通抠图
            if (!LocalModels.supported()) return detect(null)
            withModels(
                listOf(LocalModels.MODNET), "精细抠图能把头发丝、飘起来的头发抠出来，边缘更自然。",
                skip = "用普通抠图" to { detect(null) }, onCancel = { closeTool() },
            ) { (model) -> detect(model) }
        }

        private fun detect(model: File?) {
            busy("正在识别人像…") {
                val m = try {
                    Segment.person(basePreview, model)
                } finally {
                    model?.let { LocalAi.release(it) }
                }
                if (Cutout.coverage(m.data) < 0.005f) {
                    closeTool()
                    alert("没有识别到人", "抠图目前只支持有人物的照片。去掉别的东西可以用「消除」，换背景也可以在「AI 修图」里直接描述。")
                    return@busy
                }
                previewFg = withContext(Dispatchers.Default) { Matting.foregroundFull(basePreview, m) }
                mask = m
                refresh()
            }
        }

        private fun refresh() {
            val m = mask ?: return
            val e = effect
            val s = blur
            val fg = previewFg
            render { ImageIO.pixels(Segment.render(basePreview, m, e, s, fg)) }
        }

        override suspend fun result(): Bitmap? {
            val m = mask ?: return null
            return withContext(Dispatchers.Default) { Segment.render(current!!, m, effect, blur) }
        }
    }

    private inner class EraseTool : Tool("消除笔") {
        /** 这次进来后已经消除过的整图 */
        private var pending: Bitmap? = null
        private lateinit var undoStroke: View
        private lateinit var cloud: View
        private lateinit var local: View

        override fun panel(): View {
            val (sizeRow, _) = sliderRow("笔刷", 8f, 60f, 22f) { photo.brushRadius = it * resources.displayMetrics.density }
            undoStroke = button("撤销一笔") { photo.undoStroke() }
            cloud = button("云端消除") { cloudErase() }
            local = button("一键消除", primary = true) { localErase() }
            val buttons = LinearLayout(this@EditorActivity).apply {
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, dp(4), 0, 0)
                addView(undoStroke)
                addView(View(context), LinearLayout.LayoutParams(0, 1, 1f))
                addView(cloud)
                addView(View(context), LinearLayout.LayoutParams(dp(8), 1))
                addView(local)
            }
            return column(
                hint("涂抹要去掉的东西（盖住整个物体，连影子一起），双指缩放移动。「一键消除」用本地 AI 模型，免费离线；大块复杂的背景可以试试「云端消除」"),
                sizeRow,
                buttons,
            )
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
            for (v in listOf(undoStroke, cloud, local)) {
                v.isEnabled = has
            }
        }

        /** 本地 MI-GAN；32 位手机没有推理库，回到 push-pull 快速填充 */
        private fun localErase() {
            if (!LocalModels.supported()) {
                toast("这台手机不支持本地 AI 模型，改用快速消除")
                return quick()
            }
            val mb = photo.maskBitmap() ?: return
            withModel(LocalModels.MIGAN) { model ->
                busy("正在消除…") {
                    val base = pending ?: current!!
                    val r = withContext(Dispatchers.Default) {
                        val m = Cloud.alphaOf(ImageIO.scaled(mb, base.width, base.height))
                        ImageIO.toBitmap(LocalAi.inpaint(model, ImageIO.pixels(base), base.width, base.height, m), base.width, base.height)
                    }
                    show(r)
                }
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

        private fun cloudErase() {
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

    // ---------------------------------------------------------------- 用系统相册修图

    /** 交出去的文件和当时输入文件的修改时间、大小：回来时变了说明修图应用直接改了它 */
    private var editFiles: ImageIO.EditFiles? = null
    private var editStamp = 0L to 0L

    private val systemEditor = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { r -> onSystemEditDone(r.data?.data) }

    /**
     * 把当前照片交给手机自带的修图（小米、华为、OPPO 等的相册编辑器，或 Google 相册）：用它们的 AI 消除、AI 扩图等
     * 我们调不到的功能，修完再导入回来。只有一个修图应用就直接打开，多个就让用户选（不列出我们自己）。
     */
    private fun systemEdit() {
        val b = current ?: return
        busy("正在准备…") {
            val ef = withContext(Dispatchers.IO) { ImageIO.editFiles(this@EditorActivity, b) }
            // 输入只给读权限：给了写权限的话，有的修图应用（如 Markup）一打开就以写方式打开原图，把它清空
            val intent = Intent(Intent.ACTION_EDIT)
                .setDataAndType(ef.inputUri, ef.mime)
                .putExtra(MediaStore.EXTRA_OUTPUT, ef.outputUri)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            val handlers = packageManager.queryIntentActivities(intent, 0).filter { it.activityInfo.packageName != packageName }
            if (handlers.isEmpty()) {
                alert("没有可用的系统修图", "这台手机上没有能编辑图片的相册应用。")
                return@busy
            }
            // 输出文件的写权限单独授给每个候选应用
            for (h in handlers) {
                grantUriPermission(h.activityInfo.packageName, ef.outputUri, Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
            }
            editFiles = ef
            editStamp = ef.input.lastModified() to ef.input.length()
            val launch = if (handlers.size == 1) {
                intent.setClassName(handlers[0].activityInfo.packageName, handlers[0].activityInfo.name)
            } else {
                Intent.createChooser(intent, "用哪个应用修图")
                    .putExtra(Intent.EXTRA_EXCLUDE_COMPONENTS, arrayOf(ComponentName(this@EditorActivity, EditorActivity::class.java)))
            }
            try {
                systemEditor.launch(launch)
            } catch (e: ActivityNotFoundException) {
                alert("没有可用的系统修图", "这台手机上没有能编辑图片的相册应用。")
            }
        }
    }

    /**
     * 修图应用回来了。各家保存方式不同：有的把新照片的地址返回给我们，有的写进我们给的输出文件，有的直接改输入文件，
     * 有的另存到相册、什么都不告诉我们。前三种自动导入（空文件不算）；第四种请用户在相册里选刚修好的那张。
     */
    private fun onSystemEditDone(returned: Uri?) {
        val ef = editFiles ?: return
        editFiles = null
        val inputChanged = ef.input.length() > 0 && (ef.input.lastModified() != editStamp.first || ef.input.length() != editStamp.second)
        val foreign = returned?.takeIf { fromOtherApp(it) }
        when {
            foreign != null -> importEdited(foreign)
            ef.output.length() > 0 -> importEdited(Uri.fromFile(ef.output))
            inputChanged -> importEdited(Uri.fromFile(ef.input))
            else -> MaterialAlertDialogBuilder(this)
                .setTitle("导入修好的照片")
                .setMessage("如果在相册里修好并保存了，选那张照片导入回来，可以接着编辑（撤销还能回到修之前）。")
                .setPositiveButton("选择照片") { _, _ ->
                    pickPhoto { importEdited(it) }
                }
                .setNegativeButton("没有修改", null)
                .show()
        }
    }

    /**
     * 修图应用返回的地址只认别的应用提供的 content://。file:// 或指向我们自己 provider 的地址，可能是恶意应用
     * 想让我们读入自己的私有文件（我们自己的输入、输出文件按内部路径直接读，不走这里）。
     * 不比较字符串：authority 可以带「用户编号@」前缀绕过，所以带 @ 的直接拒绝，其余让系统查出它实际属于哪个应用。
     */
    private fun fromOtherApp(uri: Uri): Boolean {
        if (uri.scheme != "content") return false
        val authority = uri.authority ?: return false
        if ('@' in authority) return false
        val owner = packageManager.resolveContentProvider(authority, 0)?.packageName ?: return false
        return owner != packageName
    }

    private fun importEdited(uri: Uri) {
        busy("正在导入…") {
            val b = try {
                withContext(Dispatchers.IO) { ImageIO.decode(this@EditorActivity, uri) }
            } catch (e: SecurityException) {
                // 返回的地址我们没有读权限：改为让用户自己选
                pickPhoto { importEdited(it) }
                return@busy
            }
            commit(b)
            photo.setImage(preview)
            updateButtons()
            flashInfo("已导入修好的照片（${b.width} × ${b.height}），可以接着编辑", 4000)
        }
    }

    private fun askConfigure() {
        MaterialAlertDialogBuilder(this)
            .setTitle("需要设置 AI 服务")
            .setMessage("云端 AI 功能使用你自己的 API Key（火山引擎、Gemini 或 OpenAI 兼容接口），请先在设置里填写。推荐火山引擎：国内直连，新用户有免费额度。")
            .setPositiveButton("去设置") { _, _ -> startActivity(Intent(this, SettingsActivity::class.java)) }
            .setNegativeButton("取消", null)
            .show()
    }

    /**
     * 云端工具的公共部分：没配置 Key 时提示去设置（从设置页回来自动重建面板），生成时显示可取消的进度，结果先显示出来，点 ✓ 才采用。
     */
    private abstract inner class CloudTool(title: String, private val intro: String) : Tool(title) {
        protected var result: Bitmap? = null
        private var built = false

        /** 已经配置好服务时的面板内容 */
        abstract fun cloudPanel(): List<View>

        /** false：没配置云端服务也能用（有本地模式），照常显示面板 */
        open val needsCloud = true

        final override fun panel(): View {
            val cfg = Store.cloud()
            if (cfg == null && needsCloud) {
                built = false
                return column(
                    hint(intro + "需要先在设置里填写 API Key（推荐火山引擎，国内直连，新用户有免费额度）。"),
                    button("去设置", primary = true) { startActivity(Intent(this@EditorActivity, SettingsActivity::class.java)) },
                )
            }
            built = cfg != null
            val tail = if (cfg != null) listOf(hint(cloudNote(cfg)).apply { setPadding(0, dp(4), 0, 0) }) else emptyList()
            return column(*(cloudPanel() + tail).toTypedArray())
        }

        open fun cloudNote(cfg: Store.Cloud) = "照片会发送到 ${Store.providerName(cfg.provider)}（${cfg.model}），可在设置里更改"

        protected fun rebuild() {
            panel.removeAllViews()
            panel.addView(panel())
        }

        /** 从设置页回来：可能刚填好 Key，重建面板 */
        override fun onResume() {
            if (!built && Store.cloud() != null) {
                panel.removeAllViews()
                panel.addView(panel())
            }
        }

        protected fun generate(busyText: String, call: suspend (Store.Cloud) -> Bitmap) {
            val cfg = Store.cloud() ?: return askConfigure()
            hideKeyboard()
            busy("$busyText\n通常需要 10～60 秒", cancellable = true) {
                val r = call(cfg)
                result = r
                photo.setImage(withContext(Dispatchers.Default) { ImageIO.fit(r, PREVIEW_SIDE) })
                flashInfo("结果 ${r.width} × ${r.height}，满意就点 ✓", 4000)
            }
        }

        override suspend fun result(): Bitmap? = result

        /** 一行可横向滑动的快捷标签，点了把文字填进输入框 */
        protected fun presetRow(items: List<Pair<String, String>>, onPick: (String) -> Unit): View {
            val group = LinearLayout(this@EditorActivity).apply { gravity = Gravity.CENTER_VERTICAL }
            for ((label, text) in items) {
                group.addView(Chip(this@EditorActivity).apply {
                    this.text = label
                    setEnsureMinTouchTargetSize(false)
                    setOnClickListener { onPick(text) }
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginEnd = dp(8) })
            }
            return HorizontalScrollView(this@EditorActivity).apply {
                isHorizontalScrollBarEnabled = false
                addView(group)
            }
        }

        /** 输入框 + 生成按钮；返回 (这一行, 输入框) */
        protected fun promptRow(hintText: String, onGo: () -> Unit): Pair<View, TextInputEditText> {
            val layout = TextInputLayout(this@EditorActivity, null, com.google.android.material.R.attr.textInputOutlinedStyle).apply {
                hint = hintText
            }
            val edit = TextInputEditText(layout.context).apply {
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                maxLines = 3
                textSize = 14f
            }
            layout.addView(edit)
            val row = LinearLayout(this@EditorActivity).apply {
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, dp(6), 0, 0)
                addView(layout, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                addView(View(context), LinearLayout.LayoutParams(dp(8), 1))
                addView(button("生成", primary = true) { onGo() })
            }
            return row to edit
        }
    }

    private inner class AiTool : CloudTool("AI 修图", "「AI 修图」用一句话描述想怎么改照片，由云端图像模型完成。") {
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

        override fun cloudPanel(): List<View> {
            val recent = Store.recentPrompts().map { "↺ " + (if (it.length > 10) it.take(10) + "…" else it) to it }
            val (row, edit) = promptRow("想怎么改？例如：把天空换成晚霞") { go() }
            input = edit
            return listOf(presetRow(recent + presets) { edit.setText(it); edit.setSelection(it.length) }, row)
        }

        private fun go() {
            val prompt = input?.text?.toString()?.trim().orEmpty()
            if (prompt.isEmpty()) return toast("先写一句想怎么改")
            Store.addRecentPrompt(prompt)
            generate("AI 正在修图…") { cfg -> Cloud.edit(cfg, current!!, prompt) }
        }
    }

    // ---------------------------------------------------------------- 参考照片（换脸、换装）

    /**
     * 整个编辑页只注册一个照片选择器，选完交给这次调用传进来的回调。部分国产系统的相册对同一页面注册的第二个选择器
     * 会直接返回取消（借鉴自另一个修图项目的踩坑记录），所以参考照、导入相册修好的照片都走这一个。
     */
    private var pickRef: ((Uri) -> Unit)? = null
    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        val cb = pickRef
        pickRef = null
        if (uri != null) cb?.invoke(uri)
    }

    private fun pickPhoto(then: (Uri) -> Unit) {
        pickRef = then
        picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }

    /**
     * 「选一张参考照」的格子：点开系统照片选择器，选好显示缩略图。[onChange] 收到读好的图（长边 1024）。
     */
    private fun refSlot(label: String, initial: Bitmap? = null, onChange: (Bitmap) -> Unit): View {
        val thumb = ImageView(this).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            setBackgroundResource(R.drawable.bg_card_dark)
            clipToOutline = true
            if (initial != null) {
                setImageBitmap(initial)
            } else {
                setImageResource(R.drawable.ic_image)
                setColorFilter(ContextCompat.getColor(context, R.color.ed_text2))
                setPadding(dp(18), dp(18), dp(18), dp(18))
            }
        }
        val text = TextView(this).apply {
            this.text = label
            textSize = 14f
            setTextColor(ContextCompat.getColor(context, R.color.ed_text))
            setPadding(dp(12), 0, 0, 0)
        }
        return LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(4), 0, dp(4))
            isClickable = true
            background = ContextCompat.getDrawable(context, android.R.drawable.list_selector_background)
            addView(thumb, LinearLayout.LayoutParams(dp(64), dp(64)))
            addView(text, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            setOnClickListener {
                pickPhoto { uri ->
                    busy("正在读取照片…") {
                        val b = withContext(Dispatchers.IO) { ImageIO.fit(ImageIO.decode(this@EditorActivity, uri), 1024) }
                        thumb.setPadding(0, 0, 0, 0)
                        thumb.clearColorFilter()
                        thumb.setImageBitmap(b)
                        text.text = "已选择，点这里换一张"
                        onChange(b)
                    }
                }
            }
        }
    }

    private inner class FaceTool : CloudTool("换脸", "「换脸」把另一张照片里的脸换到这张照片的人物上，由云端图像模型完成。") {
        private var face: Bitmap? = null

        /** 0 高清（本地神经网络）、1 快速（本地网格贴合）、2 云端 AI */
        private var mode = 0
        private val cloudMode get() = mode == 2

        /** 高清模式下提供脸的人戴眼镜时怎么处理（默认去掉：不然换出来的脸会多一副半透明的眼镜框） */
        private var glasses = NeuralSwap.Glasses.REMOVE
        override val needsCloud = false

        override fun cloudNote(cfg: Store.Cloud) = if (cloudMode) super.cloudNote(cfg) else localNote()

        /** 高清换脸要下载的模型总大小，算法和下载提示框一样（按 1024 换算） */
        private val swapMb = "%.0f".format(SWAP_MODELS.sumOf { it.size } / 1048576.0)

        private fun localNote() = if (mode == 0) {
            "高清换脸用本地神经网络模型（第一次使用下载约 $swapMb MB），照片不会上传；人脸特征模型来自 InsightFace，仅限非商业使用"
        } else {
            "快速换脸不用下载模型，照片不会上传；正脸、角度相近的两张照片效果最好"
        }

        override fun enter() {
            if (Store.faceConsent) return
            MaterialAlertDialogBuilder(this@EditorActivity)
                .setTitle("使用前请确认")
                .setMessage("换脸只用于你自己的照片，或已经得到照片中本人同意的照片。不要用它冒充他人、制作误导或侵犯他人权益的内容。\n\n「本地」模式在手机上完成，照片不上传；「云端 AI」模式会把两张照片发送到你在设置里配置的服务，服务可能按自己的规则拒绝处理。")
                .setPositiveButton("我已了解") { _, _ -> Store.faceConsent = true }
                .setNegativeButton("不用了") { _, _ -> closeTool() }
                .setCancelable(false)
                .show()
        }

        override fun cloudPanel(): List<View> {
            val go = button("开始换脸", primary = true) { go() }
            val (modeRow, _) = chipRow(listOf("高清（本地）", "快速（本地）", "云端 AI"), mode) {
                mode = it
                rebuild()
            }
            return listOf(
                modeRow,
                hint("选一张正脸清楚的照片，把它的脸换到当前照片的人物上，保留当前照片的发型、表情、姿势和光线").apply { setPadding(0, dp(6), 0, dp(2)) },
            ) + (if (Store.cloud() == null && !cloudMode) listOf(hint(localNote())) else emptyList()) + listOf(
                refSlot(if (face == null) "选择提供脸的照片" else "已选择，点这里换一张", face) { face = it },
            ) + (if (mode == 0) listOf(glassesRow()) else emptyList()) + listOf(
                LinearLayout(this@EditorActivity).apply {
                    gravity = Gravity.END
                    addView(go)
                },
            )
        }

        /** 「戴眼镜时：去掉眼镜 / 保留眼镜」 */
        private fun glassesRow(): View {
            val options = NeuralSwap.Glasses.entries
            val (chips, _) = chipRow(listOf("去掉眼镜", "保留眼镜"), options.indexOf(glasses)) { glasses = options[it] }
            return LinearLayout(this@EditorActivity).apply {
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, dp(6), 0, 0)
                addView(TextView(context).apply {
                    text = "提供脸的人戴眼镜时"
                    textSize = 13f
                    setTextColor(ContextCompat.getColor(context, R.color.ed_text2))
                    setPadding(0, 0, dp(10), 0)
                })
                addView(chips)
            }
        }

        private fun go() {
            val f = face ?: return toast("先选一张提供脸的照片")
            if (cloudMode) {
                generate("AI 正在换脸…") { cfg -> Cloud.compose(cfg, current!!, f, CloudProtocol.FACE_SWAP_PROMPT) }
                return
            }
            if (mode == 1) return localSwap { FaceMesh.swap(current!!, f) }
            if (!LocalModels.supported()) return alert("不支持", "这台手机是 32 位系统，用不了高清换脸的模型，可以选「快速」或「云端 AI」。")
            withModels(
                SWAP_MODELS,
                "人脸特征模型来自 InsightFace，只允许非商业使用；换脸模型来自 FaceFusion。",
            ) { (arc, swapper, gfp) -> localSwap { FaceMesh.neuralSwap(current!!, f, arc, swapper, gfp, glasses) } }
        }

        private fun localSwap(run: suspend () -> Bitmap) {
            busy(if (mode == 0) "正在高清换脸…" else "正在换脸…") {
                val r = try {
                    run()
                } catch (e: FaceMesh.NoFace) {
                    alert("没法换脸", e.message ?: "")
                    return@busy
                }
                result = r
                photo.setImage(withContext(Dispatchers.Default) { ImageIO.fit(r, PREVIEW_SIDE) })
                flashInfo("满意就点 ✓（按住右下角对比原图）", 4000)
            }
        }
    }

    private inner class OutfitTool : CloudTool("换装", "「换装」给照片里的人换衣服：本地模式照一张衣服的照片贴到人身上；云端 AI 可以照衣服照片换，也可以直接说想穿什么。") {
        private var cloth: Bitmap? = null
        private var input: TextInputEditText? = null
        private val presets = listOf(
            "西装" to "合身的深色商务西装，白衬衫，系领带",
            "职业装" to "简洁干练的职业套装",
            "婚纱" to "白色蕾丝婚纱",
            "汉服" to "淡雅的传统汉服",
            "旗袍" to "精致的丝绸旗袍",
            "晚礼服" to "优雅的黑色晚礼服",
            "休闲装" to "白色 T 恤和蓝色牛仔裤",
            "运动装" to "运动服和运动鞋",
            "校服" to "日式学生校服",
            "冬装" to "厚实的羊毛大衣和围巾",
        )

        /** 0 本地（贴图式，只能照衣服照片换）、1 云端 AI。没配置云端服务时默认本地 */
        private var mode = if (Store.cloud() == null) 0 else 1
        override val needsCloud = false

        override fun cloudNote(cfg: Store.Cloud) = if (mode == 1) super.cloudNote(cfg) else localNote()

        private val localMb = "%.0f".format(TRYON_MODELS.sumOf { it.size } / 1048576.0)

        private fun localNote() =
            "本地换装把衣服照片变形后贴到人身上，衣服的图案、文字完全保留，照片不会上传（第一次使用下载约 $localMb MB 模型）。" +
                "没有褶皱，看起来不如云端自然；正面站立的照片、平铺或挂拍的上衣效果最好"

        override fun cloudPanel(): List<View> {
            val (modeRow, _) = chipRow(listOf("本地（快速预览）", "云端 AI"), mode) {
                mode = it
                rebuild()
            }
            if (mode == 0) {
                return listOf(
                    modeRow,
                    hint("选一张衣服的照片（平铺或挂拍、背景干净的上衣），贴到当前照片的人身上").apply { setPadding(0, dp(6), 0, dp(2)) },
                ) + (if (Store.cloud() == null) listOf(hint(localNote())) else emptyList()) + listOf(
                    refSlot(if (cloth == null) "选择衣服的照片" else "已选择，点这里换一张", cloth) { cloth = it },
                    LinearLayout(this@EditorActivity).apply {
                        gravity = Gravity.END
                        addView(button("开始换装", primary = true) { localGo() })
                    },
                )
            }
            val (row, edit) = promptRow("想穿什么？选了衣服照片可以不填") { go() }
            input = edit
            return listOf(
                modeRow,
                refSlot("选择衣服的照片（可选）", cloth) { cloth = it },
                presetRow(presets) { edit.setText(it); edit.setSelection(it.length) },
                row,
            )
        }

        private fun localGo() {
            val ref = cloth ?: return toast("先选一张衣服的照片")
            if (!LocalModels.supported()) return alert("不支持", "这台手机是 32 位系统，用不了本地换装的模型，可以选「云端 AI」。")
            withModels(TRYON_MODELS, null) { (modnet, migan, parsing, depth, harmonizer) ->
                busy("正在换装…") {
                    val r = try {
                        TryOn.run(current!!, ref, TryOn.Models(modnet, migan, parsing, depth, harmonizer))
                    } catch (e: TryOn.NoPerson) {
                        alert("没法换装", e.message ?: "")
                        return@busy
                    } catch (e: Garment.Bad) {
                        alert("没法换装", e.message ?: "")
                        return@busy
                    }
                    result = r
                    photo.setImage(withContext(Dispatchers.Default) { ImageIO.fit(r, PREVIEW_SIDE) })
                    flashInfo("满意就点 ✓（按住右下角对比原图）", 4000)
                }
            }
        }

        private fun go() {
            val desc = input?.text?.toString()?.trim().orEmpty()
            val ref = cloth
            if (ref == null && desc.isEmpty()) return toast("选一张衣服的照片，或者写想穿什么")
            val prompt = CloudProtocol.outfitPrompt(desc, hasRef = ref != null)
            generate("AI 正在换装…") { cfg ->
                if (ref != null) Cloud.compose(cfg, current!!, ref, prompt) else Cloud.edit(cfg, current!!, prompt, wrap = false)
            }
        }
    }
}
