# 生成 models/realesr-general-x4v3-int8.onnx（做法见 models/README.md「INT8 量化」）
# 需要 Python 3.12 和 onnxruntime==1.31.1 onnx numpy pillow，例如：
#   uv venv --python 3.12 /tmp/qenv && VIRTUAL_ENV=/tmp/qenv uv pip install onnxruntime==1.31.1 onnx numpy pillow
#   /tmp/qenv/bin/python scripts/quantize_esrgan.py portrait.jpg street.jpg
# 在临时目录里运行：量化工具会在当前目录留下中间文件
import os, sys, tempfile
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'models/realesr-general-x4v3.onnx')
OUT = os.path.join(ROOT, 'models/realesr-general-x4v3-int8.onnx')
# 最后一层卷积、像素重排、放大原图、相加保持浮点
TAIL = ['/body.66/Conv', '/upsampler/DepthToSpace', '/Resize', '/Add']
LAST = '/body.66/Conv'


def tiles(paths, size=192, n=16, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for p in paths:
        im = np.asarray(Image.open(p).convert('RGB'))
        for _ in range(n):
            y = rng.integers(0, im.shape[0] - size)
            x = rng.integers(0, im.shape[1] - size)
            out.append(im[y:y + size, x:x + size])
    return out


def feed(t):
    return {'input': (t.astype(np.float32) / 255).transpose(2, 0, 1)[None]}


def main(photos):
    cal = tiles(photos)
    work = tempfile.mkdtemp()
    os.chdir(work)
    pre = os.path.join(work, 'pre.onnx')
    quant_pre_process(SRC, pre, skip_symbolic_shape=True)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([feed(t) for t in cal])

        def get_next(self):
            return next(self.it, None)

    q = os.path.join(work, 'q.onnx')
    quantize_static(pre, q, Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True, nodes_to_exclude=TAIL,
                    calibrate_method=CalibrationMethod.Percentile)

    # 偏置校正：校准图上最后一层卷积输出的逐通道平均差，从它的偏置里减掉
    name = LAST + '_output_0'

    def exposed(path):
        m = onnx.load(path)
        m.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
        p = os.path.join(work, os.path.basename(path) + '.exposed.onnx')
        onnx.save(m, p)
        return ort.InferenceSession(p, providers=['CPUExecutionProvider'])

    a, b = exposed(SRC), exposed(q)
    d = np.mean([(b.run([name], feed(t))[0] - a.run([name], feed(t))[0]).mean(axis=(0, 2, 3)) for t in cal], axis=0)
    m = onnx.load(q)
    conv = next(n for n in m.graph.node if n.name == LAST)
    init = next(i for i in m.graph.initializer if i.name == conv.input[2])
    init.CopyFrom(numpy_helper.from_array((numpy_helper.to_array(init) - d).astype(np.float32), init.name))
    onnx.save(m, OUT)
    print(f'{OUT}: {os.path.getsize(OUT)} 字节，偏置最大修正 {np.abs(d).max():.4f}；改了要同步 LocalModels.ESRGAN 的校验值和大小')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('用法: quantize_esrgan.py 照片1.jpg [照片2.jpg ...]')
    main([os.path.abspath(p) for p in sys.argv[1:]])
