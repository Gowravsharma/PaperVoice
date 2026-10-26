"""
tf_shims.py - the few helpers tf_predictor.py takes from torch-importing modules of docling-ibm-models
(tableformer/common.py, utils/utils.py, data_management/transforms.py + functional.py), in numpy/cv2.
Same behaviour as the originals (docling-ibm-models, MIT licence, IBM) for the inputs the predictor gives them.
"""
import cv2
import numpy as np

import pdf_reader.vendor.tableformer.settings as s


# --- common.py
def safe_get_parameter(input_dict, index_path, default=None, required=False):
    if input_dict is None or index_path is None:
        return default
    d = input_dict
    for i in index_path[:-1]:
        if i not in d:
            if required:
                raise ValueError("Missing parameter: {}".format(i))
            return default
        d = d[i]
    last_index = index_path[-1]
    if last_index not in d:
        if required:
            raise ValueError("Missing parameter: {}".format(last_index))
        return default
    return d[last_index]


def get_prepared_data_filename(prepared_data_part, dataset_name):
    template = s.PREPARED_DATA_PARTS[prepared_data_part]
    if "<POSTFIX>" in template:
        template = template.replace("<POSTFIX>", dataset_name)
    return template


# --- utils/utils.py
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = (x[..., i] for i in range(4))
    return np.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], axis=-1)


def remove_padding(seq):
    pad_len = 0
    for x in reversed(seq):
        if x != 0:
            break
        pad_len += 1
    return (seq[:-pad_len] if pad_len > 0 else seq), pad_len


# --- data_management/transforms.py + functional.py (numpy branches; target is always None here)
class Normalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img, target=None):
        return (img.astype(np.float32) - 255.0 * np.array(self.mean)) / np.array(self.std), target


class Resize:
    def __init__(self, size, interpolation="BILINEAR"):
        self.size = size  # [w, h] as the original passes it ([448, 448])

    def __call__(self, img, target=None):
        oh, ow = self.size  # functional.resize reads a sequence size as (h, w)
        return cv2.resize(img, dsize=(int(ow), int(oh)), interpolation=cv2.INTER_LINEAR), target
