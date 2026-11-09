# 把 GFPGAN 1.4（FaceFusion 导出的 ONNX，340 MB）转成半精度（170 MB，输出和原版差 60 dB 以上，看不出来），
# 输入输出保持 float32。原模型把权重也列在了图的输入里，转换工具处理不了，先去掉。
# 环境同 quantize_esrgan.py，另需 onnxconverter-common。用法：
#   python scripts/convert_gfpgan_fp16.py gfpgan_1.4.onnx 输出目录/gfpgan_1.4_fp16.onnx
# INT8 试过：大部分权重在调制卷积的 Mul 里量化不到，只量化卷积反而 273 MB、画质掉到 30 dB，不采用。
import hashlib, os, sys
import onnx
from onnxconverter_common import float16


def main(src, out):
    m = onnx.load(src)
    inits = {i.name for i in m.graph.initializer}
    keep = [i for i in m.graph.input if i.name not in inits]
    del m.graph.input[:]
    m.graph.input.extend(keep)
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), out)
    h = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    print(f'{out}: {os.path.getsize(out)} 字节, sha256 {h}；改了要同步 LocalModels.GFPGAN 的校验值和大小')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('用法: convert_gfpgan_fp16.py gfpgan_1.4.onnx 输出.onnx')
    main(sys.argv[1], sys.argv[2])
