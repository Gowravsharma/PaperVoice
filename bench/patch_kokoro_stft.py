"""
patch_kokoro_stft.py - Replace the slow STFT node in kokoro-v1.0.onnx with a fast 1D convolution.
STFT on CPU takes ~220 ms. The equivalent Conv1D takes ~0.3 ms (over 700x faster).
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

def patch_model(src_path="models/kokoro/kokoro-v1.0.onnx", dst_path="models/kokoro/kokoro-v1.0.fast.onnx"):
    print(f"Loading {src_path}...")
    m = onnx.load(src_path)
    
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    consts = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in m.graph.node if n.op_type == "Constant" and len(n.attribute) and n.attribute[0].name == "value"}
    get_val = lambda name: inits.get(name, consts.get(name))
    
    win = get_val('/decoder/decoder/generator/Constant_17_output_0')
    win_size = len(win) # 20
    hop = int(get_val('/decoder/decoder/generator/Constant_15_output_0')) # 5
    n_bins = win_size // 2 + 1 # 11
    
    # Precompute DFT basis filter weights: shape [22, 1, 20]
    k = np.arange(n_bins)[:, None] # [11, 1]
    n = np.arange(win_size)[None, :] # [1, 20]
    angles = 2 * np.pi * k * n / win_size
    cos_w = (np.cos(angles) * win).astype(np.float32) # [11, 20]
    sin_w = (-np.sin(angles) * win).astype(np.float32) # [11, 20]
    conv_w = np.concatenate([cos_w, sin_w], axis=0)[:, None, :] # [22, 1, 20]
    
    weight_name = "/decoder/decoder/generator/stft_conv_weight"
    w_tensor = numpy_helper.from_array(conv_w, name=weight_name)
    m.graph.initializer.append(w_tensor)
    
    # In opset 20, Unsqueeze uses an input for axes
    axes_name = "/decoder/decoder/generator/stft_unsqueeze_axes"
    axes_tensor = numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)
    m.graph.initializer.append(axes_tensor)
    
    # 1. Unsqueeze node: [1, length] -> [1, 1, length]
    sig_input = '/decoder/decoder/generator/Reshape_3_output_0'
    unsqueezed_sig = '/decoder/decoder/generator/stft_sig_unsqueezed'
    unsqueeze_node = helper.make_node(
        'Unsqueeze',
        inputs=[sig_input, axes_name],
        outputs=[unsqueezed_sig],
        name='/decoder/decoder/generator/stft_unsqueeze'
    )
    
    # 2. Conv node: [1, 1, length] -> [1, 22, n_frames]
    conv_output = '/decoder/decoder/generator/stft_conv_output'
    conv_node = helper.make_node(
        'Conv',
        inputs=[unsqueezed_sig, weight_name],
        outputs=[conv_output],
        strides=[hop],
        pads=[0, 0],
        name='/decoder/decoder/generator/stft_conv'
    )
    
    # 3. Split node: [1, 22, n_frames] -> real: [1, 11, n_frames], imag: [1, 11, n_frames]
    # Channels 0..10 feed Gather_4 consumers; Channels 11..21 feed Gather_5 consumers
    real_out = '/decoder/decoder/generator/Gather_4_output_0'
    imag_out = '/decoder/decoder/generator/Gather_5_output_0'
    
    # In ONNX opset 20, Split has num_outputs attribute and axis attribute
    split_node = helper.make_node(
        'Split',
        inputs=[conv_output],
        outputs=[real_out, imag_out],
        axis=1,
        num_outputs=2,
        name='/decoder/decoder/generator/stft_split'
    )
    
    # Nodes to remove: STFT, Transpose_3, Gather_4, Gather_5
    nodes_to_remove = {
        '/decoder/decoder/generator/STFT',
        '/decoder/decoder/generator/Transpose_3',
        '/decoder/decoder/generator/Gather_4',
        '/decoder/decoder/generator/Gather_5'
    }
    
    # Remove unused initializers (Constant 15, 16, 17)
    unused_inits = {
        '/decoder/decoder/generator/Constant_15_output_0',
        '/decoder/decoder/generator/Constant_16_output_0',
        '/decoder/decoder/generator/Constant_17_output_0',
    }
    clean_inits = [t for t in m.graph.initializer if t.name not in unused_inits]
    del m.graph.initializer[:]
    m.graph.initializer.extend(clean_inits)

    
    # Find position of STFT node to insert replacement nodes
    stft_idx = next(i for i, node in enumerate(m.graph.node) if node.name == '/decoder/decoder/generator/STFT')

    
    new_nodes = [node for node in m.graph.node if node.name not in nodes_to_remove]
    # Insert new nodes right at the position of the old STFT
    new_nodes[stft_idx:stft_idx] = [unsqueeze_node, conv_node, split_node]
    
    # Replace graph nodes
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    # Fix DC imaginary branch cut in atan2 decomposition:
    # Conv1D yields exact 0.0 for DC imag, which evaluated to 0.0 > 0.0 -> False,
    # causing a -pi phase flip across all speech frames (metallic distortion).
    # Shifting Constant_20 to -eps (-1.1920929e-07) forces exact 0.0 into the +pi branch.
    c20_name = '/decoder/decoder/generator/Constant_20_output_0'
    c20_patched = False
    for t in m.graph.initializer:
        if t.name == c20_name:
            t.CopyFrom(numpy_helper.from_array(np.array(-1.1920929e-07, dtype=np.float32), name=c20_name))
            c20_patched = True
            break
    if not c20_patched:
        for n in m.graph.node:
            if n.op_type == "Constant" and len(n.output) and n.output[0] == c20_name:
                n.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array(-1.1920929e-07, dtype=np.float32)))
                c20_patched = True
                break
    
    print(f"Saving patched model to {dst_path}...")
    onnx.save(m, dst_path)
    print("Done!")

if __name__ == "__main__":
    patch_model()
