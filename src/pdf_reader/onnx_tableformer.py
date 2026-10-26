"""
onnx_tableformer.py - Docling's table structure model (TableFormer "fast") on ONNX Runtime, without PyTorch.

The network was exported by bench/export_tableformer.py into three graphs (image encoder, one tag-decoder step,
cell-box head). OnnxTableModel.predict is the decoding loop of docling-ibm-models' TableModel04_rs.predict
(MIT licence, IBM), rewritten in numpy: greedy tag decoding with the same two tag corrections, the hidden state
of every cell-starting tag kept for the box head, and the boxes of horizontally merged cells (lcel) merged.
Checked against the PyTorch model on real tables (identical tag sequences, boxes equal to ~1e-6).
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pdf_reader.models import TABLE_DIR


def session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.enable_cpu_mem_arena = False  # keep memory low between tables; the graphs are small and run rarely
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def merge_boxes(b1, b2):
    """TableModel04_rs.mergebboxes: the span of two cxcywh boxes (a cell continued by lcel tags)."""
    new_w = (b2[0] + b2[2] / 2) - (b1[0] - b1[2] / 2)
    new_h = (b2[1] + b2[3] / 2) - (b1[1] - b1[3] / 2)
    new_left = b1[0] - b1[2] / 2
    new_top = min(b2[1] - b2[3] / 2, b1[1] - b1[3] / 2)
    return np.array([new_left + new_w / 2, new_top + new_h / 2, new_w, new_h], dtype=np.float32)


class OnnxTableModel:
    def __init__(self, folder, threads=4, cell_batch=16):
        folder = Path(folder)
        self.config = json.loads((folder / "tm_config.json").read_text(encoding="utf-8"))
        self.word_map = self.config["dataset_wordmap"]["word_map_tag"]
        self.encoder = session(folder / "encoder.onnx", threads)
        self.tag_step = session(folder / "tag_step.onnx", threads)
        self.bbox = session(folder / "bbox.onnx", threads)
        self.n_layers = self.config["model"]["dec_layers"]
        self.cell_batch = cell_batch  # cells per box-head call: the attention is (cells x 784 x 512) floats

    def predict(self, imgs, max_steps, k=None):
        """imgs (1,3,448,448) float32 -> (tag sequence incl. <start>/<end>, classes (n,3), boxes (n,4) cxcywh).
        k (beam size) is accepted for the caller's signature; the original decodes greedily as well."""
        wm = self.word_map
        enc_out, memory = self.encoder.run(None, {"images": imgs.astype(np.float32)})
        tags = [wm["<start>"]]
        cache = np.zeros((self.n_layers, 0, 1, memory.shape[-1]), np.float32)
        output_tags, tag_h = [], []
        skip_next_tag, prev_tag_ucel, line_num = True, False, 0
        first_lcel, bboxes_to_merge, cur_bbox_ind, bbox_ind = True, {}, -1, 0
        cell_tags = {wm[t] for t in ("fcel", "ecel", "ched", "rhed", "srow", "nl", "ucel")}
        while len(output_tags) < max_steps:
            logits, last, cache = self.tag_step.run(None, {
                "tags": np.array(tags, np.int64)[:, None], "memory": memory, "cache": cache})
            new_tag = int(logits.argmax(1)[0])
            if line_num == 0 and new_tag == wm["xcel"]:  # (line_num is never advanced in the original either)
                new_tag = wm["lcel"]
            if prev_tag_ucel and new_tag == wm["lcel"]:
                new_tag = wm["fcel"]
            if new_tag == wm["<end>"]:
                output_tags.append(new_tag)
                tags.append(new_tag)
                break
            output_tags.append(new_tag)
            if not skip_next_tag and new_tag in cell_tags:
                tag_h.append(last)
                if first_lcel is not True:
                    bboxes_to_merge[cur_bbox_ind] = bbox_ind
                bbox_ind += 1
            if new_tag != wm["lcel"]:
                first_lcel = True
            elif first_lcel:
                tag_h.append(last)
                first_lcel = False
                cur_bbox_ind = bbox_ind
                bboxes_to_merge[cur_bbox_ind] = -1
                bbox_ind += 1
            skip_next_tag = new_tag in (wm["nl"], wm["ucel"], wm["xcel"])
            prev_tag_ucel = new_tag == wm["ucel"]
            tags.append(new_tag)

        if not tag_h:
            return tags, np.empty((0, 3), np.float32), np.empty((0, 4), np.float32)
        h = np.concatenate(tag_h, axis=0)  # (n,512)
        classes, boxes = [], []
        for i in range(0, len(h), self.cell_batch):
            c, b = self.bbox.run(None, {"enc_out": enc_out, "tag_h": h[i:i + self.cell_batch]})
            classes.append(c)
            boxes.append(b)
        classes, boxes = np.concatenate(classes), np.concatenate(boxes)

        out_cls, out_box, skip = [], [], []
        for i in range(len(boxes)):
            if i in bboxes_to_merge:
                skip.append(bboxes_to_merge[i])
                out_box.append(merge_boxes(boxes[i], boxes[bboxes_to_merge[i]]))
                out_cls.append(classes[i])
            elif i not in skip:
                out_box.append(boxes[i])
                out_cls.append(classes[i])
        return tags, np.stack(out_cls), np.stack(out_box)


def install_table_stage(model_dir=TABLE_DIR):
    """Make Docling's table stage use the ONNX model. Docling loads its table engines from a plugin list that
    also names the TableFormer-v2 and Granite-Vision engines, whose modules import torch at the top; the list is
    replaced by this one engine (it registers for the same TableStructureOptions, kind "docling_tableformer").
    Must run before the first converter is built (the plugin factory is cached)."""
    import docling.models.plugins.defaults as defaults
    from docling.datamodel.pipeline_options import TableFormerMode
    from docling.models.stages.table_structure.table_structure_model import TableStructureModel

    class OnnxTableStructureModel(TableStructureModel):
        def __init__(self, enabled, artifacts_path, options, accelerator_options, enable_remote_services=False):
            # TableStructureModel.__init__ without the PyTorch predictor
            self.options = options
            self.do_cell_matching = options.do_cell_matching
            self.mode = options.mode
            self.enabled = enabled
            self._predictor, self._threads = None, accelerator_options.num_threads
            if enabled:
                if self.mode != TableFormerMode.FAST:
                    raise ValueError("only the fast TableFormer is exported to ONNX")
                self.tm_config = json.loads((Path(model_dir) / "tm_config.json").read_text(encoding="utf-8"))
                self.tm_config["model"]["save_dir"] = str(model_dir)
                self.tm_model_type = self.tm_config["model"]["type"]
                self.scale = 2.0  # table images at 144 dpi, as in Docling

        @property
        def tf_predictor(self):
            """Loaded on the first table: Docling only reaches it for pages that have one, so documents without
            tables never hold the model (~140 MB)."""
            if self._predictor is None:
                from pdf_reader.vendor.tableformer.data_management.tf_predictor import TFPredictor
                self._predictor = TFPredictor(self.tm_config, "cpu", self._threads)
            return self._predictor

    defaults.table_structure_engines = lambda: {"table_structure_engines": [OnnxTableStructureModel]}
    return OnnxTableStructureModel
