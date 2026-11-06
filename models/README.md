# 本地模型

App 第一次用到对应功能时从这里下载（和应用内更新走同样的镜像，下载后按 SHA-256 校验，见 `core/LocalModels.kt`）。改了文件要同步改代码里的校验值和大小。

| 文件 | 用途 | 来源 | 许可 |
|---|---|---|---|
| `migan-pipeline-v2.onnx` | 消除笔「AI 消除（本地）」 | [Picsart MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN)，ONNX 来自 [andraniksargsyan/migan](https://huggingface.co/andraniksargsyan/migan) | MIT（`LICENSE-MI-GAN`） |
| `realesr-general-x4v3.onnx` | 「变清晰」放大 | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) v0.2.5.0，ONNX 来自 [vlad-grig/realesr-general-x4v3-onnx](https://huggingface.co/vlad-grig/realesr-general-x4v3-onnx) | BSD-3-Clause（`LICENSE-Real-ESRGAN`） |

MI-GAN：输入 `image` uint8 `[1,3,H,W]`（RGB）和 `mask` uint8 `[1,1,H,W]`（**0 是要补的地方**，255 保留），输出 `result` uint8 `[1,3,H,W]`，蒙版外的像素原样不动，任意尺寸。
Real-ESRGAN：输入 `input` float32 `[1,3,H,W]`（RGB 0..1），输出 `output` float32 `[1,3,4H,4W]`，任意尺寸。
