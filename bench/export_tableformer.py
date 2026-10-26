"""
export_tableformer.py - export Docling's TableFormer (fast) to ONNX, and check the export on real tables.

Usage (main env, which has torch):
    uv run python bench/export_tableformer.py --out models/tableformer_fast \
        --check papers/FASTSPEECH_2_FAST_AND_HIGH_QUALITY_END_TO.pdf:6-9 papers/tacotron-2.pdf:1-12

Three graphs (TableModel04_rs, docling-ibm-models), so the greedy decoding loop can run in numpy:
  encoder.onnx      image (1,3,448,448) -> enc_out (1,28,28,256) for the box head, memory (784,1,512) for the tags
  tag_step.onnx     all tags so far (L,1) + memory + cache (layers,L-1,1,512) -> logits, last hidden, new cache
                    (the original decoder already recomputes only the last token and caches the rest)
  bbox.onnx         enc_out + hidden states of N cells (N,512) -> classes (N,3), boxes (N,4) cxcywh; the original
                    runs the cells one by one, but each cell is independent, so they go through in batches
Copies tm_config.json next to them. The check runs Docling on the given pages with the PyTorch model,
records every table the model sees, and compares tags (must be identical) and boxes/classes.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn


class EncoderWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.enc, self.tt = m._encoder, m._tag_transformer

    def forward(self, images):
        enc_out = self.enc(images)  # (1,28,28,256)
        x = self.tt._input_filter(enc_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x.reshape(1, -1, x.shape[-1]).permute(1, 0, 2)  # (784,1,512)
        # the original passes an all-False mask (nothing masked) - the same as no mask
        memory = self.tt._encoder(x)
        return enc_out, memory


class TagStepWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.tt = m._tag_transformer

    def forward(self, tags, memory, cache):
        x = self.tt._positional_encoding(self.tt._embedding(tags))  # (L,1,512)
        out, layer_outs = x, []
        for i, layer in enumerate(self.tt._decoder.layers):
            out = layer(out, memory)  # (1,1,512): the last token only
            layer_outs.append(out)
            out = torch.cat([cache[i], out], dim=0)  # empty cache on the first step = no cache
        new_cache = torch.cat([cache, torch.stack(layer_outs, dim=0)], dim=1)
        last = out[-1]  # (1,512)
        return self.tt._fc(last), last, new_cache


class BBoxWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.d = m._bbox_decoder

    def forward(self, enc_out, tag_h):  # tag_h (N,512)
        d = self.d
        e = d._input_filter(enc_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        e = e.reshape(1, -1, e.shape[3])  # (1,P,C)
        h = d._init_h(e.mean(dim=1))  # (1,512), the same for every cell
        a = d._attention
        att = a._full_att(a._relu(a._encoder_att(e) + a._tag_decoder_att(tag_h).unsqueeze(1)
                                  + a._language_att(h).unsqueeze(1))).squeeze(2)  # (N,P)
        alpha = torch.softmax(att, dim=1)
        awe = (e * alpha.unsqueeze(2)).sum(dim=1)  # (N,C)
        awe = torch.sigmoid(d._f_beta(h)) * awe
        h = awe * h
        return d._class_embed(h), d._bbox_embed(h).sigmoid()


def load_predictor():
    from docling.models.stages.table_structure.table_structure_model import TableStructureModel
    import docling_ibm_models.tableformer.common as c
    from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
    art = Path(TableStructureModel.download_models()) / "model_artifacts" / "tableformer" / "fast"
    cfg = c.read_config(str(art / "tm_config.json"))
    cfg["model"]["save_dir"] = str(art)
    return TFPredictor(cfg, "cpu", 4), art


def export(pred, out):
    m = pred._model.eval()
    out.mkdir(parents=True, exist_ok=True)
    img = torch.rand(1, 3, 448, 448)
    with torch.no_grad():
        enc_out, memory = EncoderWrap(m).eval()(img)
        n_layers = len(m._tag_transformer._decoder.layers)
        tags = torch.tensor([[2], [5], [5]], dtype=torch.long)
        cache = torch.zeros(n_layers, 2, 1, memory.shape[-1])
        kw = dict(opset_version=17, do_constant_folding=True, dynamo=False)
        torch.onnx.export(EncoderWrap(m).eval(), (img,), out / "encoder.onnx", input_names=["images"],
                          output_names=["enc_out", "memory"], **kw)
        # the tracing exporter bakes the key length of nn.MultiheadAttention into its reshapes (the step only
        # worked at the export length), so this graph uses the torch.export-based exporter with symbolic lengths
        L, Lm1 = torch.export.Dim("L", min=1, max=1024), torch.export.Dim("Lm1", min=0, max=1023)
        torch.onnx.export(TagStepWrap(m).eval(), (tags, memory, cache), out / "tag_step.onnx",
                          input_names=["tags", "memory", "cache"], output_names=["logits", "last", "new_cache"],
                          dynamic_shapes={"tags": {0: L}, "memory": None, "cache": {1: Lm1}},
                          opset_version=18, dynamo=True, external_data=False)
        torch.onnx.export(BBoxWrap(m).eval(), (enc_out, torch.rand(3, memory.shape[-1])), out / "bbox.onnx",
                          input_names=["enc_out", "tag_h"], output_names=["classes", "boxes"],
                          dynamic_axes={"tag_h": {0: "N"}, "classes": {0: "N"}, "boxes": {0: "N"}}, **kw)


def check_graphs(pred, out):
    """Each graph against its PyTorch wrapper on the same inputs (random image, a few decoding steps)."""
    from pdf_reader.onnx_tableformer import session
    m = pred._model.eval()
    enc, step, box = (session(out / f, 4) for f in ("encoder.onnx", "tag_step.onnx", "bbox.onnx"))
    img = torch.rand(1, 3, 448, 448)
    with torch.no_grad():
        e_t, mem_t = EncoderWrap(m).eval()(img)
        e_o, mem_o = enc.run(None, {"images": img.numpy()})
        print(f"encoder: enc_out {np.abs(e_t.numpy() - e_o).max():.1e}, memory {np.abs(mem_t.numpy() - mem_o).max():.1e}")
        cache_t = torch.zeros(len(m._tag_transformer._decoder.layers), 0, 1, mem_t.shape[-1])
        cache_o, tags, worst = cache_t.numpy(), [2], 0.0
        for k in range(6):
            t = torch.tensor(tags)[:, None]
            lt, last_t, cache_t = TagStepWrap(m).eval()(t, mem_t, cache_t)
            lo, last_o, cache_o = step.run(None, {"tags": t.numpy(), "memory": mem_t.numpy(), "cache": cache_o})
            worst = max(worst, float(np.abs(lt.numpy() - lo).max()), float(np.abs(cache_t.numpy() - cache_o).max()))
            tags.append(int(lt.argmax(1)))
        print(f"tag step (6 steps): max difference {worst:.1e}")
        h = torch.rand(5, mem_t.shape[-1])
        c_t, b_t = BBoxWrap(m).eval()(e_t, h)
        c_o, b_o = box.run(None, {"enc_out": e_t.numpy(), "tag_h": h.numpy()})
        print(f"bbox: classes {np.abs(c_t.numpy() - c_o).max():.1e}, boxes {np.abs(b_t.numpy() - b_o).max():.1e}")
        # the wrappers against the original model code (batched cells vs one by one)
        cls_ref, box_ref = m._bbox_decoder.inference(e_t, [h[i:i + 1] for i in range(5)])
        print(f"bbox wrapper vs original: boxes {np.abs(box_ref.numpy() - b_t.numpy()).max():.1e}")


def check(pred, out, specs):
    """Run Docling with the PyTorch TableFormer, record each table's input and output, compare with ONNX."""
    from pdf_reader.onnx_tableformer import OnnxTableModel
    onnx_model = OnnxTableModel(out, threads=4)
    seen = []
    orig_predict = pred._model.predict

    def recording(imgs, max_steps, k, return_attention=False):
        res = orig_predict(imgs, max_steps, k)
        seen.append((imgs.numpy().copy(), res))
        return res
    pred._model.predict = recording

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    opts = PdfPipelineOptions(do_ocr=False)
    opts.table_structure_options.mode = TableFormerMode.FAST
    opts.accelerator_options = AcceleratorOptions(num_threads=4, device="cpu")
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
        pipeline_cls=StandardPdfPipeline, pipeline_options=opts)})
    conv.initialize_pipeline(InputFormat.PDF)
    # use the recording predictor inside Docling's table stage
    stage = conv._get_pipeline(InputFormat.PDF).table_model
    stage.tf_predictor = pred
    for spec in specs:
        f, pages = spec.rsplit(":", 1)
        lo, hi = (int(x) for x in pages.split("-"))
        for p in range(lo, hi + 1):
            conv.convert(f, page_range=(p, p))
    same_tags, worst_box, class_diff, cells = 0, 0.0, 0, 0
    for imgs, (seq, cls, coord) in seen:
        seq2, cls2, coord2 = onnx_model.predict(imgs, 1024)
        if seq2 == seq:
            same_tags += 1
            if len(coord):
                worst_box = max(worst_box, float(np.abs(coord.numpy() - coord2).max()))
                class_diff += int((cls.numpy().argmax(1) != cls2.argmax(1)).sum())
                cells += len(coord)
        else:
            print(f"   tags differ: torch {len(seq)} tags, onnx {len(seq2)}")
    print(f"tables: {len(seen)}; identical tag sequences: {same_tags}/{len(seen)}; cells compared: {cells}; "
          f"max box coordinate difference: {worst_box:.2e} (of 1); class changes: {class_diff}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/tableformer_fast")
    ap.add_argument("--check", nargs="*", default=[])
    ap.add_argument("--skip-export", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    pred, art = load_predictor()
    if not a.skip_export:
        export(pred, out)
        shutil.copy(art / "tm_config.json", out / "tm_config.json")
        print("exported:", ", ".join(f"{p.name} {p.stat().st_size / 2**20:.0f} MB" for p in sorted(out.glob("*.onnx"))))
    check_graphs(pred, out)
    if a.check:
        check(pred, out, a.check)


if __name__ == "__main__":
    main()
