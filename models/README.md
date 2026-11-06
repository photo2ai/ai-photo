# 本地模型

App 第一次用到对应功能时下载，先试 hf-mirror.com（模型原本在 Hugging Face，国内直连快），再试 Hugging Face 原地址，最后才用这里（GitHub 镜像国内直连只有 10～20KB/s），下载后按 SHA-256 校验，见 `core/LocalModels.kt`。这里的文件和 Hugging Face 上的是同一个；改了要同步改代码里的校验值、大小和 Hugging Face 路径。

| 文件 | 用途 | 来源 | 许可 |
|---|---|---|---|
| `migan-pipeline-v2.onnx` | 消除笔「AI 消除（本地）」 | [Picsart MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN)，ONNX 来自 [andraniksargsyan/migan](https://huggingface.co/andraniksargsyan/migan) | MIT（`LICENSE-MI-GAN`） |
| `realesr-general-x4v3.onnx` | 「变清晰」放大 | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) v0.2.5.0，ONNX 来自 [vlad-grig/realesr-general-x4v3-onnx](https://huggingface.co/vlad-grig/realesr-general-x4v3-onnx) | BSD-3-Clause（`LICENSE-Real-ESRGAN`） |

另外两个高清换脸模型太大（166 MB、265 MB），不放在这里，只从 Hugging Face 下载（[facefusion/models-3.0.0](https://huggingface.co/facefusion/models-3.0.0)，hf-mirror 优先）：

| 文件 | 用途 | 许可 |
|---|---|---|
| `arcface_w600k_r50.onnx` | 高清换脸：参考脸的身份特征 | InsightFace，**仅限非商业使用** |
| `inswapper_128_fp16.onnx` | 高清换脸：换脸网络（半精度） | InsightFace，**仅限非商业使用** |

MI-GAN：输入 `image` uint8 `[1,3,H,W]`（RGB）和 `mask` uint8 `[1,1,H,W]`（**0 是要补的地方**，255 保留），输出 `result` uint8 `[1,3,H,W]`，蒙版外的像素原样不动，任意尺寸。
Real-ESRGAN：输入 `input` float32 `[1,3,H,W]`（RGB 0..1），输出 `output` float32 `[1,3,4H,4W]`，任意尺寸。
