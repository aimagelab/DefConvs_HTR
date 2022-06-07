import cv2
import msgpack
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils import string_utils


class HwLineCollate:
    def __init__(self, width=None):
        self.width = width
    
    def __call__(self, batch):
        line_img, _, _, _ = batch[0]
        dim_c = line_img.shape[0]
        dim_h = line_img.shape[1]
        if self.width:
            dim_w = self.width
        else:
            dim_w = max({line_img.shape[2] for line_img, _, _, _ in batch})

        line_imgs = torch.zeros(len(batch), dim_c, dim_h, dim_w, dtype=torch.float32)

        labels = []
        label_lengths = []
        gt_list = []
        line_ids = []

        for i, (line_img, gt, label, line_id) in enumerate(batch):
            line_imgs[i, :, :, :line_img.shape[2]] = line_img

            labels.append(label)
            label_lengths.append(len(label))
            gt_list.append(gt)
            line_ids.append(line_id)

        labels = np.concatenate(labels)
        labels = torch.from_numpy(labels)

        return line_imgs, labels, label_lengths, gt_list, line_ids



class HwLineDataset(Dataset):
    def __init__(self, listoflists_msgpack, char2idx, root_path=".", transform=None):
        # type: (str, pandas.core.series.Series, str, ...) -> None
        self.root_path = root_path
        self.char2idx = char2idx
        with open(listoflists_msgpack, 'rb') as f:
            self.listoflists = msgpack.load(f, use_list=False)
        self.transform = transform

    def __len__(self):
        # type: () -> int
        return len(self.listoflists)

    def __getitem__(self, idx):
        # type: (int) -> Tuple[numpy.ndarray, str, numpy.ndarray]
        item = self.listoflists[idx]

        line_img = cv2.imread(os.path.join(self.root_path, item[0]))

        if line_img is None:
            print("Warning: Image is None:", os.path.join(self.root_path, item[0]))
            return None

        if self.transform:
            line_img = self.transform(line_img)

        line_id = item[0]
        gt = item[1]
        gt_label = string_utils.str2label(gt, self.char2idx)

        return line_img, gt, gt_label, line_id
