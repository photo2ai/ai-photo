# 生成高清换脸用的两个 INT8 模型（做法见 models/README.md「INT8 量化」）。1.0.12 起换脸模型换成了 HyperSwap
# （scripts/quantize_weights.py），这里的 inswapper 只留给 1.0.10～1.0.11；ArcFace 仍在用：
#   arcface_w600k_r50_int8.onnx  全部量化，Percentile 校准
#   inswapper_128_int8.onnx      先把半精度转回 fp32，只量化卷积；emap 放回最后一个 initializer（App 按这个位置读）
# 模型太大不进仓库，输出到 [输出目录]，再传到 Hugging Face（见 LocalModels.ARCFACE / INSWAPPER 的 hfPath）。
# 环境同 quantize_esrgan.py。用法：
#   python scripts/quantize_faceswap.py <arcface_w600k_r50.onnx> <inswapper_128_fp16.onnx> <输出目录> \
#       照片1.jpg:x1,y1,x2,y2,x3,y3,x4,y4,x5,y5 照片2.jpg:...
# 每张照片后面是一张脸的 5 个关键点（左眼、右眼、鼻尖、左嘴角、右嘴角，原图像素坐标），至少两个不同的人。
import hashlib, os, sys, tempfile
import numpy as np, onnx, onnxruntime as ort
from onnx import TensorProto, numpy_helper
from PIL import Image, ImageEnhance
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

T112 = np.array([38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041]).reshape(5, 2)
T128 = T112 + [8, 0]


def sess(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 4
    return ort.InferenceSession(path, o, providers=['CPUExecutionProvider'])


def similarity(src, dst):
    """最小二乘相似变换 src → dst（2×3）"""
    a_rows, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a_rows += [[x, -y, 1, 0], [y, x, 0, 1]]
        b += [u, v]
    a, c, tx, ty = np.linalg.lstsq(np.array(a_rows, float), np.array(b, float), rcond=None)[0]
    return np.array([[a, -c, tx], [c, a, ty]])


class Face:
    def __init__(self, arg):
        path, pts = arg.rsplit(':', 1)
        self.im = Image.open(path).convert('RGB')
        self.five = np.array([float(v) for v in pts.split(',')]).reshape(5, 2)

    def aligned(self, tmpl, size, rng=None, full=True):
        """
        按 5 点对齐裁出脸。rng 不为空时做数据增强：关键点抖动 ±3 像素；[full] 时再加一半概率水平翻转、亮度 ±15%
        （ArcFace 用；inswapper 的校准加了这些会把量化范围撑宽，换出的脸明显不像，只用抖动）
        """
        im, five = self.im, self.five.copy()
        if rng is not None:
            if full and rng.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                five[:, 0] = im.width - 1 - five[:, 0]
                five = five[[1, 0, 2, 4, 3]]  # 左右眼、左右嘴角对调
            five = five + rng.uniform(-3, 3, five.shape)
        m = np.linalg.inv(np.vstack([similarity(five, tmpl), [0, 0, 1]]))[:2]
        out = im.transform((size, size), Image.AFFINE, m.flatten(), Image.BILINEAR)
        if rng is not None and full:
            out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.85, 1.15))
        return np.asarray(out).astype(np.float32)


def arc_input(face112):
    return ((face112 - 127.5) / 127.5).transpose(2, 0, 1)[None].astype(np.float32)


def norm(v):
    return v / np.linalg.norm(v)


class Reader(CalibrationDataReader):
    def __init__(self, feeds):
        self.it = iter(feeds)

    def get_next(self):
        return next(self.it, None)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(arc_src, insw_src, out_dir, faces):
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp()
    os.chdir(work)  # 量化工具会在当前目录留中间文件
    rng = np.random.default_rng(1)

    # ---- ArcFace：全部量化
    arc_name = sess(arc_src).get_inputs()[0].name
    arc_cal = [{arc_name: arc_input(f.aligned(T112, 112, rng))} for f in faces for _ in range(12)]
    arc_pre = os.path.join(work, 'arc-pre.onnx')
    quant_pre_process(arc_src, arc_pre, skip_symbolic_shape=True)
    arc_out = os.path.join(out_dir, 'arcface_w600k_r50_int8.onnx')
    quantize_static(arc_pre, arc_out, Reader(arc_cal), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True, calibrate_method=CalibrationMethod.Percentile)

    # ---- inswapper：fp16 → fp32，只量化卷积
    orig = onnx.load(insw_src)
    emap_init = onnx.TensorProto()
    emap_init.CopyFrom(orig.graph.initializer[-1])
    emap = numpy_helper.to_array(emap_init).astype(np.float32)
    assert emap.shape == (512, 512), emap.shape
    m = orig
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32), init.name))
    for n in m.graph.node:
        for a in n.attribute:
            if n.op_type == 'Cast' and a.name == 'to' and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == TensorProto.FLOAT16:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))
    for v in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        if v.type.tensor_type.elem_type == TensorProto.FLOAT16:
            v.type.tensor_type.elem_type = TensorProto.FLOAT
    insw32 = os.path.join(work, 'insw-fp32.onnx')
    onnx.save(m, insw32)

    arc32 = sess(arc_src)

    def latent(face, arc=arc32, r=None):
        e = norm(arc.run(None, {arc_name: arc_input(face.aligned(T112, 112, r))})[0][0])
        return norm(e @ emap)[None].astype(np.float32)

    def feed(target, source, r):
        t = target.aligned(T128, 128, r, full=False) / 255
        return {'target': t.transpose(2, 0, 1)[None].astype(np.float32), 'source': latent(source)}

    # 目标和提供脸的人两两互换（不含自己换自己：也会撑宽量化范围），每对 6 个样本；
    # 数量别是 5 的倍数（下面每 5 个合并一次，整除时库会报错）
    pairs = [(a, b) for a in faces for b in faces if a is not b]
    insw_cal = [feed(a, b, rng) for a, b in pairs for _ in range(6)]
    if len(insw_cal) % 5 == 0:
        insw_cal = insw_cal[:-1]
    insw_pre = os.path.join(work, 'insw-pre.onnx')
    quant_pre_process(insw32, insw_pre, skip_symbolic_shape=True)
    insw_q = os.path.join(work, 'insw-q.onnx')
    quantize_static(insw_pre, insw_q, Reader(insw_cal), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=['Conv'],
                    extra_options={'CalibMaxIntermediateOutputs': 5})
    q = onnx.load(insw_q)
    keep = [i for i in q.graph.initializer if i.name != emap_init.name]
    del q.graph.initializer[:]
    q.graph.initializer.extend(keep + [emap_init])  # emap 放在最后
    insw_out = os.path.join(out_dir, 'inswapper_128_int8.onnx')
    onnx.save(q, insw_out)
    back = numpy_helper.to_array(onnx.load(insw_out).graph.initializer[-1]).astype(np.float32)
    assert np.array_equal(back, emap), 'emap 没有放在最后一个 initializer'

    # ---- 整条流程对比：原模型这一对 vs 量化这一对，用没参与校准的增强样本；「像不像」用原版 ArcFace 量
    arc8, insw16, insw8 = sess(arc_out), sess(insw_src), sess(insw_out)
    ev = np.random.default_rng(99)
    sims, psnrs = {'原模型': [], 'INT8': []}, []
    for a, b in pairs:
        for _ in range(3):
            st = ev.integers(1 << 30)
            tgt = a.aligned(T128, 128, np.random.default_rng(st)) / 255
            t_in = tgt.transpose(2, 0, 1)[None].astype(np.float32)
            ref_id = norm(arc32.run(None, {arc_name: arc_input(b.aligned(T112, 112))})[0][0])
            outs = {}
            for tag, arc, insw in (('原模型', arc32, insw16), ('INT8', arc8, insw8)):
                o = insw.run(None, {'target': t_in, 'source': latent(b, arc)})[0][0].transpose(1, 2, 0)
                o = np.clip(o, 0, 1) * 255
                outs[tag] = o
                sims[tag].append(float(norm(arc32.run(None, {arc_name: arc_input(o[0:112, 8:120])})[0][0]) @ ref_id))
            mse = np.mean((outs['原模型'] - outs['INT8']) ** 2)
            psnrs.append(10 * np.log10(255 ** 2 / mse))
    print(f"换出的脸像提供脸的人：原模型 {np.mean(sims['原模型']):.3f}，INT8 {np.mean(sims['INT8']):.3f}；"
          f"两者输出 PSNR 平均 {np.mean(psnrs):.1f} dB，最低 {np.min(psnrs):.1f} dB")
    for p in (arc_out, insw_out):
        print(f'{os.path.basename(p)}: {os.path.getsize(p)} 字节, sha256 {sha256(p)}')
    print('改了要同步 LocalModels.ARCFACE / INSWAPPER 的校验值和大小')


if __name__ == '__main__':
    if len(sys.argv) < 6:
        sys.exit('用法: quantize_faceswap.py arcface.onnx inswapper_fp16.onnx 输出目录 照片1.jpg:10个坐标 照片2.jpg:10个坐标 ...')
    main(sys.argv[1], sys.argv[2], sys.argv[3], [Face(a) for a in sys.argv[4:]])
