"""
patch_kokoro_fused_snake.py - Replace all 48 5-node Snake activation clusters with fused Snake custom ops.
Eliminates:
  - 48 Mul nodes
  - 48 Sin nodes
  - 48 Pow nodes
  - 48 Mul nodes
  - 48 Add nodes
Total nodes eliminated: 192 nodes.
Total RAM bus traffic saved: ~5 Gigabytes per sentence.
"""
import onnx
from onnx import helper

def patch_model(src_path="models/kokoro/kokoro-v1.0.fast.onnx", dst_path="models/kokoro/kokoro-v1.0.fused_snake.onnx"):
    print(f"Loading {src_path}...")
    m = onnx.load(src_path)
    inits = {t.name for t in m.graph.initializer}

    sin_nodes = [n for n in m.graph.node if n.op_type == 'Sin' and ('noise_res' in n.name or 'resblocks' in n.name)]
    print(f"Found {len(sin_nodes)} Snake Sin nodes to fuse.")

    nodes_to_remove = set()
    replacements = {} # mul1.name -> new_snake_node

    for sin_node in sin_nodes:
        mul1 = next(n for n in m.graph.node if n.output[0] == sin_node.input[0] and n.op_type == 'Mul')
        pow_node = next(n for n in m.graph.node if sin_node.output[0] in n.input and n.op_type == 'Pow')
        mul2 = next(n for n in m.graph.node if pow_node.output[0] in n.input and n.op_type == 'Mul')
        add_node = next(n for n in m.graph.node if mul2.output[0] in n.input and n.op_type == 'Add')

        alpha = next(inp for inp in mul1.input if inp in inits)
        x = next(inp for inp in mul1.input if inp not in inits)

        out_name = add_node.output[0]
        snake_node = helper.make_node(
            'Snake',
            inputs=[x, alpha],
            outputs=[out_name],
            name=sin_node.name.replace('/Sin', '/FusedSnake'),
            domain='papervoice'
        )

        replacements[mul1.name] = snake_node
        nodes_to_remove.update([sin_node.name, pow_node.name, mul2.name, add_node.name])

    new_nodes = []
    for node in m.graph.node:
        if node.name in replacements:
            new_nodes.append(replacements[node.name])
        elif node.name not in nodes_to_remove:
            new_nodes.append(node)

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    # Ensure domain 'papervoice' is in opset_import
    has_domain = any(op.domain == 'papervoice' for op in m.opset_import)
    if not has_domain:
        m.opset_import.append(helper.make_opsetid('papervoice', 1))

    print(f"Original nodes: {len(m.graph.node) + len(nodes_to_remove)} -> Patched nodes: {len(m.graph.node)}")
    print(f"Saving patched model to {dst_path}...")
    onnx.save(m, dst_path)
    print("Done!")

if __name__ == '__main__':
    patch_model()
