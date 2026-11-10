# 算 ArcFace 身份特征里的「眼镜」方向，生成 app/.../core/GlassesDirection.kt。
# 做法：在不戴眼镜的人脸上画眼镜（不同形状、粗细、颜色），同一张脸画前画后特征的差取平均——个人差异互相抵消，剩下眼镜。
# 换脸时把提供者特征在这个方向上的投影推回「不戴眼镜的人的平均」，换出来的脸就不会带一副淡淡的眼镜；反过来加一点，
# 就会画出完整的眼镜。用的人脸是 StyleGAN 生成的虚构人脸（Hugging Face 数据集 TheKernel01/140k-Real-and-Fake-Faces，
# test 集前 200 行里标为 fake 的），关键点用 MediaPipe 人脸网格（和 App 的 ML Kit 同一套编号）。
# 环境同 quantize_esrgan.py，另需 mediapipe。用法：
#   python scripts/glasses_direction.py <arcface_w600k_r50_int8.onnx> <工作目录>
import glob, json, os, sys, urllib.request
import numpy as np, onnxruntime as ort
from PIL import Image, ImageDraw
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'app/src/main/java/com/photo2ai/aiphoto/core/GlassesDirection.kt')
ROWS = 'https://datasets-server.huggingface.co/rows?dataset=TheKernel01/140k-Real-and-Fake-Faces&config=default&split=test&offset={}&length=100'
LANDMARKER = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
# 这批虚构人脸里本来就戴眼镜 / 墨镜的（编号是 fake 图的序号），不参与「画眼镜」，另作验证用
REAL_GLASSES = [17, 44, 46, 47, 52, 54, 55, 60, 65, 69, 79, 91]
SUNGLASSES = [7, 72, 92]
T112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])
LEFT = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]


def similarity(src, dst):
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a += [[x, -y, 1, 0], [y, x, 0, 1]]
        b += [u, v]
    p, q, tx, ty = np.linalg.lstsq(np.array(a, float), np.array(b, float), rcond=None)[0]
    return np.array([[p, -q, tx], [q, p, ty]])


def download(work):
    os.makedirs(work, exist_ok=True)
    urls = []
    for off in (0, 100):
        rows = json.load(urllib.request.urlopen(ROWS.format(off), timeout=60))['rows']
        urls += [r['row']['image']['src'] for r in rows if r['row']['label'] == 1]
    for i, u in enumerate(urls):
        p = os.path.join(work, f'f{i}.jpg')
        if not os.path.exists(p):
            urllib.request.urlretrieve(u, p)
    task = os.path.join(work, 'face_landmarker.task')
    if not os.path.exists(task):
        urllib.request.urlretrieve(LANDMARKER, task)
    return task


def five_points(lm, path):
    img = mp.Image.create_from_file(path)
    r = lm.detect(img)
    if not r.face_landmarks:
        return None
    p = np.array([[l.x * img.width, l.y * img.height] for l in r.face_landmarks[0]])
    eyes = sorted([p[LEFT].mean(0), p[RIGHT].mean(0)], key=lambda q: q[0])
    mouth = sorted([p[61], p[291]], key=lambda q: q[0])
    return np.array([eyes[0], eyes[1], p[1], mouth[0], mouth[1]])


def aligned112(path, f5):
    m = np.linalg.inv(np.vstack([similarity(f5, T112), [0, 0, 1]]))[:2]
    return np.asarray(Image.open(path).convert('RGB').transform((112, 112), Image.AFFINE, m.flatten(), Image.BILINEAR)).astype(np.float32)


def draw_glasses(face, rng):
    """在 112×112 对齐脸上画一副眼镜（放大 4 倍画再缩小，边缘平滑）"""
    s4 = 4
    im = Image.fromarray(np.uint8(face)).resize((112 * s4, 112 * s4), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    w, h, t = rng.uniform(26, 34), rng.uniform(17, 24), rng.uniform(1.2, 3.0)
    col = tuple(int(v) for v in (rng.uniform(15, 70, 3) if rng.random() < 0.7 else rng.uniform(110, 170, 3)))
    shape = rng.choice(['rect', 'ellipse', 'round'])
    dy = rng.uniform(-1, 3)
    for x, y in ((38.29, 51.70), (73.53, 51.50)):
        box = [(x - w / 2) * s4, (y - h / 2 + dy) * s4, (x + w / 2) * s4, (y + h / 2 + dy) * s4]
        if shape == 'ellipse':
            d.ellipse(box, outline=col, width=int(t * s4))
        elif shape == 'round':
            d.rounded_rectangle(box, radius=int(h * 0.35 * s4), outline=col, width=int(t * s4))
        else:
            d.rectangle(box, outline=col, width=int(t * s4))
    yb = (51.6 - h * 0.25 + dy) * s4
    d.line([((38.29 + w / 2) * s4, yb), ((73.53 - w / 2) * s4, yb)], fill=col, width=int(t * s4))
    d.line([((38.29 - w / 2) * s4, (51.7 - h * 0.3 + dy) * s4), (0, (50 + dy) * s4)], fill=col, width=int(t * s4))
    d.line([((73.53 + w / 2) * s4, (51.5 - h * 0.3 + dy) * s4), (112 * s4, (50 + dy) * s4)], fill=col, width=int(t * s4))
    return np.asarray(im.resize((112, 112), Image.LANCZOS)).astype(np.float32)


def main(arcface, work):
    task = download(work)
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=task), num_faces=1))
    arc = ort.InferenceSession(arcface, providers=['CPUExecutionProvider'])

    def emb(face):
        e = arc.run(None, {arc.get_inputs()[0].name: ((face - 127.5) / 127.5).transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        return e / np.linalg.norm(e)

    rng, diffs, plain = np.random.default_rng(0), [], {}
    for path in sorted(glob.glob(os.path.join(work, 'f*.jpg'))):
        k = int(os.path.basename(path)[1:-4])
        if k in SUNGLASSES:
            continue
        f5 = five_points(lm, path)
        if f5 is None:
            continue
        face = aligned112(path, f5)
        plain[k] = emb(face)
        if k in REAL_GLASSES:
            continue
        for _ in range(3):
            diffs.append(emb(draw_glasses(face, rng)) - plain[k])
    g = np.mean(diffs, 0)
    g /= np.linalg.norm(g)
    no_glasses = [float(e @ g) for k, e in plain.items() if k not in REAL_GLASSES]
    with_glasses = [float(e @ g) for k, e in plain.items() if k in REAL_GLASSES]
    mu0 = float(np.mean(no_glasses))
    print(f'{len(diffs)} 对画前画后；验证：真戴眼镜的投影 {min(with_glasses):+.3f}～{max(with_glasses):+.3f}，'
          f'不戴的 {min(no_glasses):+.3f}～{max(no_glasses):+.3f}，平均 {mu0:+.4f}')
    vals = ',\n'.join('        ' + ', '.join(f'{v:.7f}f' for v in g[i:i + 8]) for i in range(0, 512, 8))
    open(OUT, 'w').write(f'''package com.photo2ai.aiphoto.core

/**
 * ArcFace（arcface_w600k_r50_int8.onnx）身份特征里的「眼镜」方向，由 scripts/glasses_direction.py 生成，别手改。
 * 在不戴眼镜的虚构人脸上画眼镜，同一张脸画前画后特征的差取平均：个人差异互相抵消，剩下的就是「戴眼镜」。
 * 验证：真戴眼镜的人在这个方向上的投影 {min(with_glasses):+.3f}～{max(with_glasses):+.3f}，不戴的 {min(no_glasses):+.3f}～{max(no_glasses):+.3f}。
 */
object GlassesDirection {{
    /** 不戴眼镜的人在这个方向上的平均投影 */
    const val NO_GLASSES_MEAN = {mu0:.4f}f

    /** 单位向量，512 维 */
    val VECTOR = floatArrayOf(
{vals},
    )
}}
''')
    print('已生成', OUT)
    return g


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('用法: glasses_direction.py arcface_w600k_r50_int8.onnx 工作目录')
    g = main(sys.argv[1], sys.argv[2])
