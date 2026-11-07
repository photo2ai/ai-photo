package com.photo2ai.aiphoto.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.GridLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.photo2ai.aiphoto.BuildConfig
import com.photo2ai.aiphoto.R

class MainActivity : AppCompatActivity() {
    /** 选完照片后直接打开的工具（点首页下面的格子时） */
    private var pendingTool: String? = null
    private var pendingEffect: String? = null

    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        if (uri != null) {
            startActivity(
                Intent(this, EditorActivity::class.java)
                    .setData(uri)
                    .putExtra(EditorActivity.EXTRA_TOOL, pendingTool)
                    .putExtra(EditorActivity.EXTRA_EFFECT, pendingEffect)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION),
            )
        }
    }

    /** [color] 是图标颜色，底色用它的浅色版本 */
    private class Tile(val tool: String, val icon: Int, val title: String, val color: Int, val effect: String? = null)

    private val tiles = listOf(
        Tile(EditorActivity.TOOL_ENHANCE, R.drawable.ic_magic, "智能增强", 0xFFFF8A3D.toInt()),
        Tile(EditorActivity.TOOL_UPSCALE, R.drawable.ic_hd, "变清晰", 0xFF3D8BFF.toInt()),
        Tile(EditorActivity.TOOL_ERASE, R.drawable.ic_eraser, "消除笔", 0xFF2FB862.toInt()),
        Tile(EditorActivity.TOOL_CUTOUT, R.drawable.ic_person, "人像抠图", 0xFF7B61FF.toInt()),
        Tile(EditorActivity.TOOL_CUTOUT, R.drawable.ic_idphoto, "证件照", 0xFF22B8A7.toInt(), effect = "WHITE"),
        Tile(EditorActivity.TOOL_CUTOUT, R.drawable.ic_bokeh, "背景虚化", 0xFF5AA9FF.toInt(), effect = "BLUR"),
        Tile(EditorActivity.TOOL_AI, R.drawable.ic_sparkle, "AI 修图", 0xFF9B5CFF.toInt()),
        Tile(EditorActivity.TOOL_ADJUST, R.drawable.ic_tune, "调色", 0xFFF5A623.toInt()),
        Tile(EditorActivity.TOOL_FILTER, R.drawable.ic_filter, "滤镜", 0xFFFF4D80.toInt()),
        Tile(EditorActivity.TOOL_CROP, R.drawable.ic_crop, "裁剪", 0xFF3BB4F2.toInt()),
    )

    private class Promo(val tool: String, val icon: Int, val title: String, val sub: String, val from: Int, val to: Int)

    private val promos = listOf(
        Promo(EditorActivity.TOOL_FACE, R.drawable.ic_face, "AI 换脸", "换成参考照片的脸", 0xFFFF6B9A.toInt(), 0xFFFFA26B.toInt()),
        Promo(EditorActivity.TOOL_OUTFIT, R.drawable.ic_shirt, "AI 换装", "西装汉服婚纱随心换", 0xFF8B6CFF.toInt(), 0xFF5AA9FF.toInt()),
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        findViewById<View>(R.id.root).applySystemInsets()
        pendingTool = savedInstanceState?.getString("tool")
        pendingEffect = savedInstanceState?.getString("effect")
        findViewById<View>(R.id.pick).setOnClickListener { pick(null, null) }
        findViewById<View>(R.id.settings).setOnClickListener { startActivity(Intent(this, SettingsActivity::class.java)) }
        findViewById<TextView>(R.id.footer).text = "版本 " + BuildConfig.VERSION_NAME

        val promoRow = findViewById<LinearLayout>(R.id.promos)
        promos.forEachIndexed { i, p ->
            val card = FrameLayout(this).apply {
                background = GradientDrawable(GradientDrawable.Orientation.TL_BR, intArrayOf(p.from, p.to)).apply { cornerRadius = dp(16).toFloat() }
                foreground = ContextCompat.getDrawable(context, android.R.drawable.list_selector_background)
                isClickable = true
                setOnClickListener { pick(p.tool, null) }
                addView(ImageView(context).apply {
                    setImageResource(p.icon)
                    setColorFilter(Color.WHITE)
                    alpha = 0.25f
                }, FrameLayout.LayoutParams(dp(52), dp(52), Gravity.END or Gravity.BOTTOM).apply { setMargins(0, 0, dp(10), dp(8)) })
                addView(LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(dp(14), dp(14), dp(14), dp(12))
                    addView(TextView(context).apply {
                        text = p.title
                        textSize = 18f
                        setTextColor(Color.WHITE)
                        paint.isFakeBoldText = true
                    })
                    addView(TextView(context).apply {
                        text = p.sub
                        textSize = 12f
                        setTextColor(0xE6FFFFFF.toInt())
                        setPadding(0, dp(2), 0, 0)
                    })
                    addView(TextView(context).apply {
                        text = "去试试 ›"
                        textSize = 12f
                        setTextColor(p.from)
                        paint.isFakeBoldText = true
                        background = ContextCompat.getDrawable(context, R.drawable.bg_white_pill)
                        setPadding(dp(10), dp(3), dp(10), dp(3))
                    }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(10) })
                })
            }
            promoRow.addView(card, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f).apply {
                if (i > 0) marginStart = dp(12)
            })
        }

        val grid = findViewById<GridLayout>(R.id.tiles)
        for (t in tiles) {
            val item = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                setPadding(0, dp(10), 0, dp(10))
                background = ContextCompat.getDrawable(context, android.R.drawable.list_selector_background)
                isClickable = true
                setOnClickListener { pick(t.tool, t.effect) }
                addView(ImageView(context).apply {
                    setImageResource(t.icon)
                    setColorFilter(t.color)
                    setPadding(dp(10), dp(10), dp(10), dp(10))
                    background = GradientDrawable().apply {
                        shape = GradientDrawable.OVAL
                        setColor((t.color and 0xFFFFFF) or 0x24000000)
                    }
                }, LinearLayout.LayoutParams(dp(44), dp(44)))
                addView(TextView(context).apply {
                    text = t.title
                    textSize = 12f
                    gravity = Gravity.CENTER
                    setTextColor(ContextCompat.getColor(context, R.color.text))
                    setPadding(0, dp(8), 0, 0)
                })
            }
            grid.addView(item, GridLayout.LayoutParams(GridLayout.spec(GridLayout.UNDEFINED), GridLayout.spec(GridLayout.UNDEFINED, 1f)).apply {
                width = 0
                setGravity(Gravity.FILL_HORIZONTAL)
            })
        }
        UpdateUi.autoCheck(this)
    }

    override fun onResume() {
        super.onResume()
        UpdateUi.resume(this)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString("tool", pendingTool)
        outState.putString("effect", pendingEffect)
    }

    private fun pick(tool: String?, effect: String?) {
        pendingTool = tool
        pendingEffect = effect
        picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }
}
