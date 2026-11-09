"""
verify_stft_conv.py - Verify that STFT(N=20, hop=5, win=hann) is mathematically
identical to a 1D convolution with precomputed cosine/sine filter weights.
"""
import numpy as np
import onnx
import onnxruntime as rt

def create_stft_conv_weights(win_size=20, onesided=True):
    n_fft = win_size
    n_bins = (n_fft // 2) + 1 if onesided else n_fft # 11
    
    # Hann window
    # In kokoro, window is:
    n = np.arange(win_size)
    # PyTorch hann_window(20, periodic=True) or matching Constant_17
    win = 0.5 * (1 - np.cos(2 * np.pi * n / win_size))
    
    # Build DFT basis
    # k in [0, 1, ..., n_bins-1]
    # n in [0, 1, ..., win_size-1]
    k = np.arange(n_bins)[:, None] # [11, 1]
    n = np.arange(win_size)[None, :] # [1, 20]
    
    angles = 2 * np.pi * k * n / n_fft
    cos_basis = np.cos(angles) * win # [11, 20]
    sin_basis = -np.sin(angles) * win # [11, 20]
    
    # Conv1D weights shape: [out_channels, in_channels, kernel_size]
    # out_channels = 11 (real) + 11 (imag) = 22
    weights = np.stack([cos_basis, sin_basis], axis=1) # [11, 2, 20]
    return cos_basis, sin_basis

def test_equivalence():
    m = onnx.load("models/kokoro/kokoro-v1.0.onnx", load_external_data=False)
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in m.graph.initializer}
    consts = {n.output[0]: onnx.numpy_helper.to_array(n.attribute[0].t) for n in m.graph.node if n.op_type == "Constant" and len(n.attribute) and n.attribute[0].name == "value"}
    get_val = lambda name: inits.get(name, consts.get(name))
    
    win = get_val('/decoder/decoder/generator/Constant_17_output_0')
    win_size = len(win) # 20
    hop = int(get_val('/decoder/decoder/generator/Constant_15_output_0')) # 5
    n_bins = win_size // 2 + 1 # 11
    
    # Random signal matching input shape [1, 168020]
    np.random.seed(42)
    sig_len = 1000 # test smaller length first
    x = np.random.randn(1, sig_len).astype(np.float32)
    
    # 1. Run via ORT STFT node
    # Build minimal ONNX model with STFT
    from onnx import helper, TensorProto
    x_val = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, None])
    out_val = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, None, n_bins, 2])
    
    step_init = helper.make_tensor('step', TensorProto.INT64, [], [hop])
    win_init = helper.make_tensor('win', TensorProto.FLOAT, [win_size], win.tolist())
    len_init = helper.make_tensor('len', TensorProto.INT64, [], [win_size])
    
    stft_node = helper.make_node(
        'STFT',
        inputs=['X', 'step', 'win', 'len'],
        outputs=['Y'],
        onesided=1
    )
    graph = helper.make_graph([stft_node], 'test_stft', [x_val], [out_val], [step_init, win_init, len_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    
    sess = rt.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    ort_stft_out = sess.run(None, {'X': x})[0] # [1, n_frames, 11, 2]
    
    # 2. Run via 1D convolution
    # Kernel for real: cos * win, for imag: -sin * win
    k = np.arange(n_bins)[:, None]
    n = np.arange(win_size)[None, :]
    angles = 2 * np.pi * k * n / win_size
    cos_w = (np.cos(angles) * win).astype(np.float32) # [11, 20]
    sin_w = (-np.sin(angles) * win).astype(np.float32) # [11, 20]
    conv_w = np.concatenate([cos_w, sin_w], axis=0)[:, None, :] # [22, 1, 20]
    
    # Build minimal Conv1D model
    conv_x_val = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, None])
    conv_out_val = helper.make_tensor_value_info('Y_conv', TensorProto.FLOAT, [1, 22, None])
    w_init = helper.make_tensor('W', TensorProto.FLOAT, [22, 1, win_size], conv_w.flatten().tolist())
    
    conv_node = helper.make_node(
        'Conv',
        inputs=['X', 'W'],
        outputs=['Y_conv'],
        strides=[hop],
        pads=[0, 0] # no pad, signal is already padded in generator
    )
    c_graph = helper.make_graph([conv_node], 'test_conv', [conv_x_val], [conv_out_val], [w_init])
    c_model = helper.make_model(c_graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)

    
    c_sess = rt.InferenceSession(c_model.SerializeToString(), providers=['CPUExecutionProvider'])
    conv_out = c_sess.run(None, {'X': x[:, None, :]})[0] # [1, 22, n_frames]
    
    # Transpose STFT out: [1, n_frames, 11, 2] -> real: [1, 11, n_frames], imag: [1, 11, n_frames]
    real_stft = ort_stft_out[0, :, :, 0].T[None, :, :]
    imag_stft = ort_stft_out[0, :, :, 1].T[None, :, :]
    combined_stft = np.concatenate([real_stft, imag_stft], axis=1)
    
    max_diff = np.max(np.abs(combined_stft - conv_out))
    mean_diff = np.mean(np.abs(combined_stft - conv_out))
    print(f"STFT shape: {combined_stft.shape}, Conv shape: {conv_out.shape}")
    print(f"Max absolute difference: {max_diff:.8e}")
    print(f"Mean absolute difference: {mean_diff:.8e}")
    if max_diff < 5e-5:

        print("PASS: STFT and Conv1D outputs are mathematically equivalent within float32 precision!\n")
    else:
        print("FAIL: Differences found\n")
        
    # Benchmark on full audio length: 168,020 samples
    print("=== Speed Benchmark (sig_len = 168,020 samples, 33,601 frames) ===")
    import time
    x_full = np.random.randn(1, 168020).astype(np.float32)
    
    # Warm-up
    sess.run(None, {'X': x_full})
    c_sess.run(None, {'X': x_full[:, None, :]})
    
    # Time STFT
    t0 = time.perf_counter()
    for _ in range(10):
        _ = sess.run(None, {'X': x_full})
    t_stft = (time.perf_counter() - t0) / 10
    
    # Time Conv1D
    t0 = time.perf_counter()
    for _ in range(10):
        _ = c_sess.run(None, {'X': x_full[:, None, :]})
    t_conv = (time.perf_counter() - t0) / 10
    
    print(f"STFT execution time:   {t_stft * 1000:6.2f} ms")
    print(f"Conv1D execution time: {t_conv * 1000:6.2f} ms")
    print(f"Speedup:               {t_stft / t_conv:6.2f}x faster!")

if __name__ == "__main__":
    test_equivalence()

