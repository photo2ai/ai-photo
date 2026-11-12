package com.photo2ai.aiphoto.core

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.annotation.DrawableRes
import com.photo2ai.aiphoto.R
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.zip.ZipFile

/**
 * 换装的常用衣服：缩略图随 APK（res/drawable-nodpi/outfit_*.webp），衣服大图和关键点标注在按需下载的衣服包
 * [LocalModels.OUTFITS] 里（每件 <id>.webp + <id>.json）。生成和来源见 scripts/garments/README.md。
 */
object Outfits {
    class Item(val id: String, val name: String, val category: String, @DrawableRes val thumb: Int)

    /** 顺序就是面板里的顺序；id 和 scripts/garments/pack.py 的 IDS 一致 */
    val ITEMS = listOf(
        Item("tee_white", "白T恤", "日常", R.drawable.outfit_tee_white),
        Item("tee_black", "黑T恤", "日常", R.drawable.outfit_tee_black),
        Item("tee_stripe", "海魂衫", "日常", R.drawable.outfit_tee_stripe),
        Item("polo", "Polo衫", "日常", R.drawable.outfit_polo),
        Item("shirt_white", "白衬衫", "日常", R.drawable.outfit_shirt_white),
        Item("shirt_plaid", "格子衬衫", "日常", R.drawable.outfit_shirt_plaid),
        Item("hoodie", "卫衣", "日常", R.drawable.outfit_hoodie),
        Item("sweater", "毛衣", "日常", R.drawable.outfit_sweater),
        Item("sport", "运动服", "日常", R.drawable.outfit_sport),
        Item("cardigan", "开衫", "外套", R.drawable.outfit_cardigan),
        Item("denim_jacket", "牛仔外套", "外套", R.drawable.outfit_denim_jacket),
        Item("bomber", "夹克", "外套", R.drawable.outfit_bomber),
        Item("down_jacket", "羽绒服", "外套", R.drawable.outfit_down_jacket),
        Item("trench", "风衣", "外套", R.drawable.outfit_trench),
        Item("suit", "西装", "正装", R.drawable.outfit_suit),
        Item("jk", "JK制服", "正装", R.drawable.outfit_jk),
        Item("dress_black", "小黑裙", "裙装", R.drawable.outfit_dress_black),
        Item("gown", "晚礼服", "裙装", R.drawable.outfit_gown),
        Item("qipao", "红旗袍", "裙装", R.drawable.outfit_qipao),
        Item("qipao_green", "绿旗袍", "裙装", R.drawable.outfit_qipao_green),
        Item("wedding", "蕾丝婚纱", "婚礼", R.drawable.outfit_wedding),
        Item("wedding_satin", "缎面婚纱", "婚礼", R.drawable.outfit_wedding_satin),
        Item("xiuhe", "秀禾服", "婚礼", R.drawable.outfit_xiuhe),
        Item("groom", "新郎礼服", "婚礼", R.drawable.outfit_groom),
        Item("hanfu", "汉服", "古装", R.drawable.outfit_hanfu),
        Item("mamianqun", "马面裙", "古装", R.drawable.outfit_mamianqun),
        Item("yuanlingpao", "圆领袍", "古装", R.drawable.outfit_yuanlingpao),
        Item("tangzhuang", "唐装", "古装", R.drawable.outfit_tangzhuang),
        Item("dragon", "龙袍", "古装", R.drawable.outfit_dragon),
        Item("qing", "清装", "古装", R.drawable.outfit_qing),
        Item("kimono", "和服", "异国", R.drawable.outfit_kimono),
        Item("hanbok", "韩服", "异国", R.drawable.outfit_hanbok),
    )

    val CATEGORIES = ITEMS.map { it.category }.distinct()

    /** 从衣服包里读出一件：带透明背景的衣服图和标注 */
    fun load(zip: File, id: String): Pair<Bitmap, Garment.Annotation> = ZipFile(zip).use { z ->
        fun bytes(name: String) = z.getInputStream(z.getEntry(name) ?: throw IOException("衣服包里没有 $name")).use { it.readBytes() }
        val img = bytes("$id.webp")
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val bmp = BitmapFactory.decodeByteArray(img, 0, img.size, opts) ?: throw IOException("衣服图读不出来：$id")
        bmp to annotation(JSONObject(bytes("$id.json").decodeToString()))
    }

    /** 标注 JSON（scripts/garments/README.md 里的格式）→ [Garment.Annotation] */
    fun annotation(j: JSONObject): Garment.Annotation {
        fun v(k: String) = j.optJSONArray(k)?.let { V(it.getDouble(0).toFloat(), it.getDouble(1).toFloat()) }
        fun req(k: String) = v(k) ?: throw IOException("衣服标注缺少 $k")
        fun poly(k: String) = j.optJSONArray(k)?.let { a ->
            List(a.length()) { i -> a.getJSONArray(i).let { V(it.getDouble(0).toFloat(), it.getDouble(1).toFloat()) } }
        }
        fun num(k: String) = if (j.has(k)) j.getDouble(k).toFloat() else null
        return Garment.Annotation(
            req("SL"), req("SR"), req("AL"), req("AR"), req("HL"), req("HR"), req("N"),
            wl = v("WL"), wr = v("WR"), sleeveL = poly("sleeveL"), sleeveR = poly("sleeveR"),
            long = j.optBoolean("long"), shoulder = num("shoulder"), ease = num("ease"), sleeveSqueeze = num("sleeveSqueeze") ?: 1f,
        )
    }
}
