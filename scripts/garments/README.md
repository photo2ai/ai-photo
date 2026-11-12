# 换装的常用衣服

App 换装面板里的 32 件常用衣服，按日常 / 外套 / 正装 / 裙装 / 婚礼 / 古装 / 异国分类（`core/Outfits.kt`）。缩略图随 APK（`res/drawable-nodpi/outfit_*.webp`），大图和关键点标注打成 `models/outfits_v1.zip`（约 2.4 MB），第一次用时下载（`LocalModels.OUTFITS`，先 hf-mirror，Hugging Face 上是 `Lakes0001/ai-photo-models/outfits_v1.zip`）。

来源：

- 28 件是 AI 生成的白底 / 灰底平铺照（2026-11，Figma 的 AI 出图，模型 gpt-image），原图缩到长边 1024 放在 `ai_src/`，由 `ai.py` 抠图和标注。
- 4 件是博物馆开放馆藏（CC0），由 `museum.py` 下载、抠图和标注：

| id | 名字 | 来源 |
|---|---|---|
| `qipao` | 红旗袍 | Smithsonian NMAAHC [2007.3.685](https://nmaahc.si.edu/object/nmaahc_2007.3.685) |
| `wedding_satin` | 缎面婚纱 | Smithsonian NMAAHC [2007.3.824](https://nmaahc.si.edu/object/nmaahc_2007.3.824) |
| `dragon` | 龙袍 | The Met [65642](https://www.metmuseum.org/art/collection/search/65642) |
| `qing` | 清装（女常服袍） | Art Institute of Chicago（经 [rawpixel](https://www.rawpixel.com/image/9001266)） |

## 重新生成

```bash
python scripts/garments/ai.py /tmp/g/ai             # AI 生成的 28 件：抠图、写标注
python scripts/garments/museum.py /tmp/g/museum     # 博物馆的 4 件（museum.py 里的原图会下载）
python scripts/garments/pack.py /tmp/g/ai /tmp/g/museum models/outfits_v1.zip app/src/main/res
```

依赖 numpy、scipy、opencv-python、Pillow。`pack.py` 打印 zip 的大小和 SHA-256，改了要同步 `LocalModels.OUTFITS`；发布后再改要换文件名（`outfits_v2.zip`），老版本 App 还认旧文件。同样的输入打出来的 zip 每次一样。`pack.py` 的 `IDS` 要和 `Outfits.ITEMS` 一致。

加一件 AI 衣服：生成正面平铺、左右对称、整件入画、纯白或浅灰背景的图，缩到长边 1024 存成 `ai_src/<id>.jpg`；跑一次 `ai.py` 后把抠图叠上网格读坐标，在 `ai.py` 的 `ITEMS` 里写中线 `cx` 和左半边的关键点（右半边镜像），再跑一次把标注画回图上核对（躯干、袖子分别染色）。

## 标注格式（`<id>.json`，衣服图的像素坐标）

| 键 | 含义 |
|---|---|
| `SL` `SR` | 左右肩点（画面左右）；袖子从这里接出去 |
| `AL` `AR` | 左右腋下 |
| `HL` `HR` | 躯干下摆两角（长款就是裙摆 / 袍摆两角） |
| `N` | 领口中点（衣服顶上） |
| `WL` `WR` | 可选，腰：网格多一行，贴身的旗袍、婚纱靠它收到身体宽度 |
| `sleeveL` `sleeveR` | 可选，袖子轮廓多边形，和衣服蒙版取交集（多边形里的碎块只留最大一块当袖子）；没有就是无袖 |
| `long` | 长款：盖住下半身，光照只按胯部以上估，露出来的旧裤子也补掉 |
| `shoulder` | 可选，衣服上多宽（像素）对应人的肩宽。平铺的袍服肩线在袖子接缝处，比人肩宽得多，靠它定衣长、袖长 |
| `ease` | 可选，腋下、腰两行按「身体宽度 × ease」贴体（可收可放）；腋下收窄时下摆按同样比例收 |
| `sleeveSqueeze` | 可选，袖子横向宽度比例（宽袖垂下来叠在一起，正面看窄） |

`ai.py` 里长袖的袖子轮廓不是手写的：腋下以下逐行找衣身和袖子之间的背景缝，沿真实的缝走（衣身侧边是弯的，用直线会把鼓出来的衣身算进袖子，袖子截面量宽，贴上去会甩出一片）。

自动识别（`Garment.parse`）只认平铺的上衣；婚纱、旗袍、袍服在电脑原型上直接贴会把立领当肩、垂袖认错、在胯部截断，所以内置衣服都带标注。没收的：碎花泡泡袖连衣裙（泡泡袖不管拆成袖子还是并进衣身，贴上去都像翅膀）。
