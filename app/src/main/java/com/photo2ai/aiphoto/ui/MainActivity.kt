package com.photo2ai.aiphoto.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
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

    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        if (uri != null) {
            startActivity(
                Intent(this, EditorActivity::class.java)
                    .setData(uri)
                    .putExtra(EditorActivity.EXTRA_TOOL, pendingTool)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION),
            )
        }
    }

    private class Tile(val tool: String, val icon: Int, val title: String, val sub: String)

    private val tiles = listOf(
        Tile(EditorActivity.TOOL_ENHANCE, R.drawable.ic_magic, "智能增强", "一键调好曝光和色彩"),
        Tile(EditorActivity.TOOL_CUTOUT, R.drawable.ic_person, "人像抠图", "换背景 · 虚化 · 证件照"),
        Tile(EditorActivity.TOOL_ERASE, R.drawable.ic_eraser, "涂抹消除", "去路人、杂物、瑕疵"),
        Tile(EditorActivity.TOOL_AI, R.drawable.ic_sparkle, "AI 修图", "说一句话改照片"),
        Tile(EditorActivity.TOOL_FILTER, R.drawable.ic_filter, "滤镜", "胶片 · 日系 · 黑白"),
        Tile(EditorActivity.TOOL_CROP, R.drawable.ic_crop, "裁剪旋转", "常用比例一键裁"),
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        findViewById<android.view.View>(R.id.root).applySystemInsets()
        pendingTool = savedInstanceState?.getString("tool")
        findViewById<android.view.View>(R.id.pick).setOnClickListener { pick(null) }
        findViewById<android.view.View>(R.id.settings).setOnClickListener { startActivity(Intent(this, SettingsActivity::class.java)) }
        findViewById<TextView>(R.id.footer).text = "版本 " + BuildConfig.VERSION_NAME
        val grid = findViewById<GridLayout>(R.id.tiles)
        val gap = dp(6)
        for (t in tiles) {
            val card = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setBackgroundResource(R.drawable.bg_card)
                setPadding(dp(16), dp(16), dp(16), dp(16))
                isClickable = true
                isFocusable = true
                foreground = ContextCompat.getDrawable(context, android.R.drawable.list_selector_background)
                setOnClickListener { pick(t.tool) }
                addView(ImageView(context).apply {
                    setImageResource(t.icon)
                    setColorFilter(ContextCompat.getColor(context, R.color.accent))
                }, LinearLayout.LayoutParams(dp(28), dp(28)))
                addView(TextView(context).apply {
                    text = t.title
                    textSize = 16f
                    setTextColor(ContextCompat.getColor(context, R.color.text))
                    paint.isFakeBoldText = true
                    setPadding(0, dp(12), 0, 0)
                })
                addView(TextView(context).apply {
                    text = t.sub
                    textSize = 12f
                    setTextColor(ContextCompat.getColor(context, R.color.text2))
                    setPadding(0, dp(2), 0, 0)
                })
            }
            grid.addView(card, GridLayout.LayoutParams(GridLayout.spec(GridLayout.UNDEFINED), GridLayout.spec(GridLayout.UNDEFINED, 1f)).apply {
                width = 0
                setMargins(gap, gap, gap, gap)
                setGravity(Gravity.FILL)
            })
        }
        UpdateUi.autoCheck(this)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString("tool", pendingTool)
    }

    private fun pick(tool: String?) {
        pendingTool = tool
        picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }
}
