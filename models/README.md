# 本地模型

App 第一次用到对应功能时下载，先试 hf-mirror.com（模型原本在 Hugging Face，国内直连快），再试 Hugging Face 原地址，最后才用这里（GitHub 镜像国内直连只有 10～20KB/s），下载后按 SHA-256 校验，见 `core/LocalModels.kt`。这里的文件和 Hugging Face 上的是同一个（我们自己量化的版本除外，只在这里）；改了要同步改代码里的校验值、大小和 Hugging Face 路径。

| 文件 | 用途 | 来源 | 许可 |
|---|---|---|---|
| `migan-pipeline-v2.onnx` | 消除笔「AI 消除（本地）」 | [Picsart MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN)，ONNX 来自 [andraniksargsyan/migan](https://huggingface.co/andraniksargsyan/migan) | MIT（`LICENSE-MI-GAN`） |
| `realesr-general-x4v3-int8.onnx` | 「变清晰」放大、高清换脸里把脸放大到 512（1.0.9 起） | 下面这个模型的 INT8 量化版，见下文 | BSD-3-Clause（`LICENSE-Real-ESRGAN`） |
| `realesr-general-x4v3.onnx` | 1.0.8 及以前的「变清晰」（老版本还会来这里下，不能删） | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) v0.2.5.0，ONNX 来自 [vlad-grig/realesr-general-x4v3-onnx](https://huggingface.co/vlad-grig/realesr-general-x4v3-onnx) | BSD-3-Clause（`LICENSE-Real-ESRGAN`） |

另外两个高清换脸模型太大，不放在这里，只从 Hugging Face 下载。1.0.9 及以前用原版（[facefusion/models-3.0.0](https://huggingface.co/facefusion/models-3.0.0)），1.0.10 起用我们的 INT8 版（见下文「INT8 量化」）：

| 文件 | 用途 | 许可 |
|---|---|---|
| `arcface_w600k_r50.onnx` | 高清换脸：参考脸的身份特征 | InsightFace，**仅限非商业使用** |
| `inswapper_128_fp16.onnx` | 高清换脸：换脸网络（半精度） | InsightFace，**仅限非商业使用** |

MI-GAN：输入 `image` uint8 `[1,3,H,W]`（RGB）和 `mask` uint8 `[1,1,H,W]`（**0 是要补的地方**，255 保留），输出 `result` uint8 `[1,3,H,W]`，蒙版外的像素原样不动，任意尺寸。
Real-ESRGAN：输入 `input` float32 `[1,3,H,W]`（RGB 0..1），输出 `output` float32 `[1,3,4H,4W]`，任意尺寸。

## INT8 量化（1.0.9）

`realesr-general-x4v3-int8.onnx` 由浮点版用 ONNX Runtime 1.31.1 静态量化得到：QDQ 格式，权重逐通道 int8、激活 uint8，Percentile 校准（照片 `portrait.jpg`、`street.jpg` 上随机取 32 块 192×192）；最后一层卷积、像素重排、加回放大原图这几步保持浮点；再做偏置校正：在校准图上统计量化版和原模型最后一层卷积输出的逐通道平均差，从这层的偏置里减掉（平均亮度偏差从 +0.9 降到 ±0.3）。

实测（Apple M 系列 CPU，4 线程，192×192 块）：4.87 MB → 1.41 MB，每块 296 ms → 约 130 ms；和浮点版输出的 PSNR，照片平均 43.7 dB（最低 38.8），文字类细线、色块约 33 dB，满对比度的反色棋盘格最差（19～31 dB）。校准集里加锐利图案会让照片掉到 35 dB，前几层保持浮点也没用，所以只用照片校准。

试过但没采用：
- MI-GAN（消除）：只量化卷积，28 MB → 11 MB、快约 1.5 倍，但补出来的区域发灰、有斑块（以原照片为准，涂抹区 PSNR 31.4 → 27.4 dB）；颜色相关的层保持浮点、补图区低频颜色用 push-pull 校正都只能补回不到 1 dB

### 高清换脸的两个模型（ArcFace 1.0.10 起使用；inswapper 只用到 1.0.11，1.0.12 起换成 HyperSwap，见下文）

`scripts/quantize_faceswap.py` 生成（用法见脚本开头；两次运行结果相同），放在 Hugging Face 仓库 [Lakes0001/ai-photo-models](https://huggingface.co/Lakes0001/ai-photo-models)（App 经 hf-mirror 下载）：

| 文件 | 体积 | SHA-256 | 做法 |
|---|---|---|---|
| `arcface_w600k_r50_int8.onnx` | 43,977,929（原 174 MB） | `43deedbae06600d32b569046c29bb9aee94f34d1be79a88f033e653383f96d8d` | 全部量化，Percentile 校准，校准用对齐后的脸加翻转、亮度变化 |
| `inswapper_128_int8.onnx` | 177,474,191（原 277 MB） | `1a2c7ad36b25d214c4cab89acf109db00829894aafa6710845c9e554e9473bc8` | 半精度转回 fp32 后只量化卷积；emap 放回最后一个 initializer（App 按这个位置读） |

效果（电脑 CPU；两张测试照片互换，评估样本带校准时没见过的翻转、亮度变化）：
- ArcFace：和原模型同一张脸的特征相似度 0.992，快约 3 倍
- 整条流程：换出的脸像提供脸的人，原模型这一对 0.909、INT8 这一对 0.900；两者输出 32 dB；inswapper 快约 2.7 倍
- 踩过的坑：inswapper 全部量化（含身份注入部分）139 MB，但胡子出斑块、嘴唇发虚（0.872）；校准时加翻转、亮度变化或「自己换自己」，量化范围被撑宽，只剩 0.76。所以只量化卷积、只用两人互换 + 关键点抖动校准
- 局限：校准和评估只有两个人（thispersondoesnotexist 现在返回网页，拿不到更多人脸），换到别的脸上的效果没有量过

换模型时：重新生成后传到上面的仓库（同名覆盖会让老版本 App 校验失败，所以改名再传），改 `LocalModels.ARCFACE` / `INSWAPPER` 的文件名、校验值、大小、`hfPath`、`replaces`，以及 `EditorActivity` 里写的下载大小。单测 `FaceSwapModelTest.neuralSwapInt8MatchesFloat` 在本机 `/tmp/aiphoto-models/q/` 有这两个文件时会核对 emap 和换脸结果。

### 人脸修复：GFPGAN 1.4 半精度（1.0.11 起）

`gfpgan_1.4_fp16.onnx`（170,261,734 字节，SHA-256 `a94d1ae4353e9fee4f7dae2639bf9b3311188eab3d40b9ec8cc821e8aa42db3c`）由 [facefusion/models-3.0.0](https://huggingface.co/facefusion/models-3.0.0) 的 `gfpgan_1.4.onnx`（340 MB）经 `scripts/convert_gfpgan_fp16.py` 转成半精度，放在同一个 Hugging Face 仓库。许可 Apache 2.0（[TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)）。输入 `input` float32 `[1,3,512,512]`（RGB，值域 -1..1，按 FFHQ 模板对齐），输出同形状。

选型（电脑上两张测试照片互换，同样的换脸结果后面接不同的修复）：
- 不修复：inswapper 只有 128×128，放大后发糊，胡子上有白色噪点
- GPEN-BFR-256（76 MB，半精度 38 MB，电脑上 87 ms）：清楚很多，但残留噪点和胡子阴影；许可只许非商业
- GFPGAN 1.4（电脑上 1.4 秒）：最自然，眼睛、牙齿干净，胡子痕迹基本消失。半精度和原版差 62 dB；只量化卷积的 INT8 反而 273 MB（大部分权重在调制卷积的 Mul 里量化不到）、画质掉到 30 dB，不采用

内存：电脑上半精度版会话默认（全部图优化）建好后 849 MB、跑一次峰值 1059 MB，图优化会把权重提前展开成一份 fp32；关掉图优化后 362 / 801 MB，速度不变，所以 App 里这个模型关掉图优化（`Accel.options`）。它也不测速、固定用 CPU：权重是按风格向量现算的，NNAPI、XNNPACK 帮不上，测速要先后建几个会话，2.5 GB 内存的模拟器上会被系统杀掉。inswapper 也不测速：CPU 上 1 秒多，模拟器上几种加速的结果都和 CPU 对不上；第一次换脸时它的测速（4 次载入 177 MB 模型）紧接着 GFPGAN，App 在 2.5 GB 的模拟器上被杀（占用 1.6 GB），去掉后首次使用正常。

### 换脸模型：HyperSwap 1c（1.0.12 起，取代 inswapper）

`hyperswap_1c_256_w8.onnx`（202,061,901 字节，SHA-256 `702ec8e5dd1b45a89abe366c5815262e49789ba770353e29c5043de036e5b949`）由 [facefusion/models-3.3.0](https://huggingface.co/facefusion/models-3.3.0) 的 `hyperswap_1c_256.onnx`（403 MB，本身就是半精度）经 `scripts/convert_hyperswap.py` 生成：卷积 / 反卷积 / 全连接的权重按输出通道量化成 int8，前接 DequantizeLinear，激活值不量化。和原版差 51～52 dB、速度一样；App 里关掉内存复用规划（电脑上跑一次的峰值 1093 → 705 MB）。模型由 FaceFusion Labs 训练，**没有写明模型许可**（FaceFusion 代码是 OpenRAIL-AS）。输入 `target` float32 `[1,3,256,256]`（RGB，-1..1，按 arcface_128 模板对齐）、`source` `[1,512]`（归一化的 ArcFace 特征），输出 `output`（-1..1）和 `mask`（不用）。

选型（电脑上两张测试照片互换，同样的对齐、贴回；「像不像」用 ArcFace 打分，天然偏向用它训练的 inswapper）：

| 模型 | 只换脸 | 像提供者（女→男 / 男→女） | 耗时 | 体积 |
|---|---|---|---|---|
| inswapper 128 | 噪点、胡子痕迹，必须修复 | 0.89 / 0.90 | 0.4～0.6 秒 | 554 MB |
| HyperSwap 1a / 1b / 1c 256 | 干净，带过提供者的妆容；男→女时换得不够彻底 | 0.82～0.83 / 0.75～0.79 | 0.5～1.1 秒 | 403 MB |
| AlphaFace 256 | 不错，有胡子残留 | 0.84 / 0.86 | 6.6～8.7 秒 | 556 MB |
| GHOST 256 | 噪点多 | 0.80 / 0.69 | 0.5～0.7 秒 | 515 + 22 MB |

缩小看时 HyperSwap 只换脸就够干净，但原尺寸（脸宽 500～700 像素）下 256 的输出偏软、牙齿有瑕疵，接上 GFPGAN 后最好：女→男没有胡子痕迹、更像本人，比 inswapper + GFPGAN 好。换脸强度（FaceFusion 的 face_swapper_weight）调高不会更像（0.75 不变，1.0 略降）。压缩：静态 INT8（激活也量化）只有 23～25 dB，且反卷积量化不到、仍有 320～357 MB，不采用。

`inswapper_128_int8.onnx` 仍留在 Hugging Face 仓库给 1.0.10～1.0.11 下载；新版启动时会删掉手机上的旧 inswapper 文件。
