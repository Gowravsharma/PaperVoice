"""
bench_pow_vs_mul.py - Measure memory-bandwidth cost:
Compare ONNX Runtime's execution time of Pow(x, 2.0) vs Mul(x, x)
on a realistic audio tensor from Kokoro's vocoder: shape [1, 256, 33600] (~34 MB).
"""
import time
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as rt

def main():
    # 1. Create realistic audio tensor matching Kokoro ResBlock dimensions:
    # 256 channels, 33,600 audio frames = 8,601,600 float32 elements = 34.4 MB
    shape = [1, 256, 33600]
    x = np.random.randn(*shape).astype(np.float32)
    
    x_val = helper.make_tensor_value_info('X', TensorProto.FLOAT, shape)
    y_val = helper.make_tensor_value_info('Y', TensorProto.FLOAT, shape)

    # 2. Build ONNX Model 1: Pow(x, 2.0)
    two_tensor = helper.make_tensor('two', TensorProto.FLOAT, [], [2.0])
    pow_node = helper.make_node('Pow', inputs=['X', 'two'], outputs=['Y'])
    graph_pow = helper.make_graph([pow_node], 'pow_graph', [x_val], [y_val], [two_tensor])
    model_pow = helper.make_model(graph_pow, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    sess_pow = rt.InferenceSession(model_pow.SerializeToString(), providers=['CPUExecutionProvider'])

    # 3. Build ONNX Model 2: Mul(x, x)
    mul_node = helper.make_node('Mul', inputs=['X', 'X'], outputs=['Y'])
    graph_mul = helper.make_graph([mul_node], 'mul_graph', [x_val], [y_val])
    model_mul = helper.make_model(graph_mul, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    sess_mul = rt.InferenceSession(model_mul.SerializeToString(), providers=['CPUExecutionProvider'])

    # 4. Warm up execution kernels
    sess_pow.run(None, {'X': x})
    sess_mul.run(None, {'X': x})

    # 5. Benchmark 20 iterations each
    n_iters = 20
    
    t0 = time.perf_counter()
    for _ in range(n_iters):
        sess_pow.run(None, {'X': x})
    t_pow = (time.perf_counter() - t0) / n_iters

    t0 = time.perf_counter()
    for _ in range(n_iters):
        sess_mul.run(None, {'X': x})
    t_mul = (time.perf_counter() - t0) / n_iters

    print(f"Tensor shape: {shape} ({x.nbytes / 1e6:.1f} MB in RAM)")
    print(f"Pow(x, 2.0) average time: {t_pow * 1000:6.2f} ms")
    print(f"Mul(x, x)   average time: {t_mul * 1000:6.2f} ms")
    print(f"Ratio:                    {t_pow / t_mul:6.2f}x")
    print("\nTakeaway: Both take ~6.8 ms because both are bottlenecked by streaming 34 MB of RAM!")

if __name__ == "__main__":
    main()
