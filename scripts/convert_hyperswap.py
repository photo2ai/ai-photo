# 把 HyperSwap（FaceFusion 导出的 ONNX，权重半精度 403 MB）转成「只量化权重」的 INT8（约 200 MB）。
# 卷积 / 反卷积 / 全连接的权重按输出通道对称量化成 int8，前面接 DequantizeLinear 现场还原成 fp32，激活值不量化：
# 画质接近原版，运行时内存里也只有 int8 那一份。其余小权重（偏置、归一化）存 fp32。
# 静态量化（激活值也量化）试过：和原版只差 23～25 dB，而且反卷积量化不到，体积 320～357 MB，不采用。
# 环境同 quantize_esrgan.py。用法：
#   python scripts/convert_hyperswap.py hyperswap_1c_256.onnx 输出目录/hyperswap_1c_256_w8.onnx
import hashlib, os, sys
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper


def output_axis(op, attrs, w):
    """权重里「输出通道」那一维：Conv 是 0；ConvTranspose 是 1；Gemm 看 transB（B 是 [K,N] 时输出在 1）"""
    if op == 'Conv':
        return 0
    if op == 'ConvTranspose':
        return 1
    return 0 if attrs.get('transB', 0) else 1


def main(src, out):
    m = onnx.load(src)
    # 半精度权重先转 fp32（Cast 节点、常量、类型声明一起改），再统一处理
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
    inits = {i.name: i for i in m.graph.initializer}
    keep_inputs = [i for i in m.graph.input if i.name not in inits]
    del m.graph.input[:]
    m.graph.input.extend(keep_inputs)

    new_nodes, done, count = [], {}, 0
    for n in m.graph.node:
        if n.op_type in ('Conv', 'ConvTranspose', 'Gemm') and len(n.input) > 1 and n.input[1] in inits:
            wname = n.input[1]
            if wname not in done:
                w = numpy_helper.to_array(inits[wname]).astype(np.float32)
                axis = output_axis(n.op_type, {a.name: helper.get_attribute_value(a) for a in n.attribute}, w)
                red = tuple(i for i in range(w.ndim) if i != axis)
                scale = np.maximum(np.abs(w).max(axis=red), 1e-12) / 127.0
                shape = [1] * w.ndim
                shape[axis] = -1
                q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
                inits[wname].CopyFrom(numpy_helper.from_array(q, wname + '_q'))
                m.graph.initializer.append(numpy_helper.from_array(scale.astype(np.float32), wname + '_scale'))
                new_nodes.append(helper.make_node('DequantizeLinear', [wname + '_q', wname + '_scale'], [wname + '_dq'], axis=axis,
                                                  name=wname + '_DequantizeLinear'))
                done[wname] = wname + '_dq'
                count += int(w.size)
            n.input[1] = done[wname]
        new_nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    # DequantizeLinear 的 axis 要求 opset >= 13
    for o in m.opset_import:
        if o.domain in ('', 'ai.onnx') and o.version < 13:
            o.version = 13
    onnx.checker.check_model(m)
    onnx.save(m, out)
    h = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    print(f'量化了 {len(done)} 个权重（{count / 1e6:.0f}M 参数）。{out}: {os.path.getsize(out)} 字节, sha256 {h}；'
          f'改了要同步 LocalModels.HYPERSWAP 的校验值和大小')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('用法: convert_hyperswap.py hyperswap_1c_256.onnx 输出.onnx')
    main(sys.argv[1], sys.argv[2])
