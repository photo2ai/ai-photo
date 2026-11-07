# 本地模型

App 第一次用到对应功能时下载，先试 hf-mirror.com（模型原本在 Hugging Face，国内直连快），再试 Hugging Face 原地址，最后才用这里（GitHub 镜像国内直连只有 10～20KB/s），下载后按 SHA-256 校验，见 `core/LocalModels.kt`。这里的文件和 Hugging Face 上的是同一个（我们自己量化的版本除外，只在这里）；改了要同步改代码里的校验值、大小和 Hugging Face 路径。

| 文件 | 用途 | 来源 | 许可 |
|---|---|---|---|
| `migan-pipeline-v2.onnx` | 消除笔「AI 消除（本地）」 | [Picsart MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN)，ONNX 来自 [andraniksargsyan/migan](https://huggingface.co/andraniksargsyan/migan) | MIT（`LICENSE-MI-GAN`） |
| `realesr-general-x4v3-int8.onnx` | 「变清晰」放大、高清换脸里把脸放大到 512（1.0.9 起） | 下面这个模型的 INT8 量化版，见下文 | BSD-3-Clause（`LICENSE-Real-ESRGAN`） |
| `realesr-general-x4v3.onnx` | 1.0.8 及以前的「变清晰」（老版本还会来这里下，不能删） | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) v0.2.5.0，ONNX 来自 [vlad-grig/realesr-general-x4v3-onnx](https://huggingface.co/vlad-grig/realesr-general-x4v3-onnx) | BSD-3-Clause（`LICENSE-Real-ESRGAN`） |

另外两个高清换脸模型太大（166 MB、265 MB），不放在这里，只从 Hugging Face 下载（[facefusion/models-3.0.0](https://huggingface.co/facefusion/models-3.0.0)，hf-mirror 优先）：

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
- ArcFace（换脸身份特征）：全量化 174 MB → 44 MB、快 3 倍，和浮点版特征的余弦相似度平均 0.98（最低 0.96）
- inswapper（换脸）：先把半精度转回 fp32 再量化。按 5 点对齐的脸上比较：全部量化 139 MB、快 3.5 倍，但胡子出斑块、嘴唇发虚，换出的脸像提供者的程度 0.913 → 0.872；只量化卷积 176 MB、快 2.7 倍，和浮点版肉眼看不出差别（0.906，31 dB）
- 两个换脸模型的量化版没接进 App：它们现在从 facefusion 的 Hugging Face 仓库下载，量化版没有国内能快速下载的地方放（GitHub 单文件上限 100 MB，镜像每秒十几 KB）；InsightFace 只许非商业使用，改过的模型公开放出去算不算再分发要另外确认
