"""
verify_fused_snake.py - Verify numerical accuracy and latency of the Fused Snake Custom Operator.
Compares:
  Unfused: Mul -> Sin -> Pow(..., 2.0) -> Mul -> Add
  Fused:   Snake(X, Alpha) via tools/snake_op.dll
"""
import time
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

def create_unfused_model():
    # Inputs: X [1, 256, 33601], Alpha [1, 256, 1], InvAlpha [1, 256, 1], Two [1]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 256, 33601])
    Alpha = helper.make_tensor_value_info('Alpha', TensorProto.FLOAT, [1, 256, 1])
    InvAlpha = helper.make_tensor_value_info('InvAlpha', TensorProto.FLOAT, [1, 256, 1])
    Two = helper.make_tensor_value_info('Two', TensorProto.FLOAT, [1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 256, 33601])

    node_mul1 = helper.make_node('Mul', ['X', 'Alpha'], ['t1'])
    node_sin  = helper.make_node('Sin', ['t1'], ['t2'])
    node_pow  = helper.make_node('Pow', ['t2', 'Two'], ['t3'])
    node_mul2 = helper.make_node('Mul', ['t3', 'InvAlpha'], ['t4'])
    node_add  = helper.make_node('Add', ['X', 't4'], ['Y'])

    graph = helper.make_graph([node_mul1, node_sin, node_pow, node_mul2, node_add], 'unfused_snake', [X, Alpha, InvAlpha, Two], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    return model.SerializeToString()

def create_fused_model():
    # Inputs: X [1, 256, 33601], Alpha [1, 256, 1]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 256, 33601])
    Alpha = helper.make_tensor_value_info('Alpha', TensorProto.FLOAT, [1, 256, 1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 256, 33601])

    node_snake = helper.make_node('Snake', ['X', 'Alpha'], ['Y'], domain='papervoice')

    graph = helper.make_graph([node_snake], 'fused_snake', [X, Alpha], [Y])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid('', 17),
        helper.make_opsetid('papervoice', 1)
    ], ir_version=9)
    return model.SerializeToString()

def main():
    print("Initializing test tensors (Batch=1, Channels=256, Time=33,601 -> 34.4 MB)...")
    np.random.seed(42)
    x = np.random.randn(1, 256, 33601).astype(np.float32)
    alpha = np.random.uniform(0.5, 2.0, (1, 256, 1)).astype(np.float32)
    inv_alpha = (1.0 / alpha).astype(np.float32)
    two = np.array([2.0], dtype=np.float32)

    # 1. Unfused Session
    unfused_bytes = create_unfused_model()
    sess_unfused = ort.InferenceSession(unfused_bytes, providers=['CPUExecutionProvider'])
    out_unfused = sess_unfused.run(None, {'X': x, 'Alpha': alpha, 'InvAlpha': inv_alpha, 'Two': two})[0]

    # 2. Fused Session with snake_op.dll
    so = ort.SessionOptions()
    so.register_custom_ops_library("tools/snake_op.dll")
    fused_bytes = create_fused_model()
    sess_fused = ort.InferenceSession(fused_bytes, so, providers=['CPUExecutionProvider'])
    out_fused = sess_fused.run(None, {'X': x, 'Alpha': alpha})[0]

    # Accuracy verification
    max_diff = np.max(np.abs(out_unfused - out_fused))
    mean_diff = np.mean(np.abs(out_unfused - out_fused))
    corr = np.corrcoef(out_unfused.ravel(), out_fused.ravel())[0, 1]

    print("\n--- Numerical Verification ---")
    print(f"Max Absolute Difference:  {max_diff:.6e}")
    print(f"Mean Absolute Difference: {mean_diff:.6e}")
    print(f"Pearson Correlation:      {corr:.8f}")

    # Latency benchmark (10 iterations)
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        sess_unfused.run(None, {'X': x, 'Alpha': alpha, 'InvAlpha': inv_alpha, 'Two': two})
    t_unfused = (time.perf_counter() - t0) / N * 1000.0

    t0 = time.perf_counter()
    for _ in range(N):
        sess_fused.run(None, {'X': x, 'Alpha': alpha})
    t_fused = (time.perf_counter() - t0) / N * 1000.0

    print("\n--- Latency Benchmark (Per Layer, 34.4 MB Tensor) ---")
    print(f"Unfused (5 ops, 172 MB RAM traffic): {t_unfused:.2f} ms")
    print(f"Fused   (1 op,   68 MB RAM traffic):  {t_fused:.2f} ms")
    print(f"Speedup:                              {t_unfused / t_fused:.2f}x faster!")
    print(f"Projected savings across 50 layers:   {(t_unfused - t_fused) * 50:.1f} ms per sentence")

if __name__ == '__main__':
    main()
