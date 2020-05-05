import os
import os.path as osp

from PIL import Image
import numpy as np
import torch

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import motmetrics as mm
mm.lap.default_solver = 'lap'


class COSMOSTestDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.transforms = ToTensor()
        img_dir_path = osp.join(root, 'img1')
        gt_txt_path = osp.join(root, 'gt', 'gt.txt')

        self._img_paths = {}
        img_names = []
        for img_name in os.listdir(img_dir_path):
            img_names.append(img_name)
        img_names.sort()
        for i in range(len(img_names)):
            self._img_paths[i + 1] = osp.join(img_dir_path, img_names[i])

        self.anns = {i: {} for i in range(1, len(self._img_paths) + 1)}
        self.vises = {i: {} for i in range(1, len(self._img_paths) + 1)}
        with open(gt_txt_path, 'r') as file:
            data = file.readlines()
            for l in data:
                line = l.split(',')
                self.anns[int(line[0])][int(line[1])] = np.array(
                    [int(line[2]), int(line[3]), int(line[2]) + int(line[4]), int(line[3]) + int(line[5])],
                    dtype='float32')
                self.vises[int(line[0])][int(line[1])] = 1.0

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx + 1]
        img = Image.open(frame_img_path).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.zeros((1, 4))
        sample['img_path'] = frame_img_path
        sample['gt'] = self.anns[idx + 1]
        sample['vis'] = self.vises[idx + 1]
        return sample

    def __len__(self):
        return len(self._img_paths)

    def __str__(self):
        return osp.split(self.root)[1]


def get_mot_accum_new(results, seq):
    mot_accum = mm.MOTAccumulator(auto_id=True)
    for i, data in enumerate(seq):
        gt = data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)
            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])
        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4])
        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack((track_boxes[:, 0],
                                    track_boxes[:, 1],
                                    track_boxes[:, 2] - track_boxes[:, 0],
                                    track_boxes[:, 3] - track_boxes[:, 1]),
                                    axis=1)
        else:
            track_boxes = np.array([])
        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)
        mot_accum.update(
            gt_ids,
            track_ids,
            distance)
        if i == len(seq)-1:
            break
    return mot_accum


def get_mot_accum_new_2(results, seq):
    mot_accum = mm.MOTAccumulator(auto_id=True)
    for i, data in enumerate(seq):
        gt = data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)
            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])
        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4])
        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack((track_boxes[:, 0],
                                    track_boxes[:, 1],
                                    track_boxes[:, 2] - track_boxes[:, 0],
                                    track_boxes[:, 3] - track_boxes[:, 1]),
                                    axis=1)
        else:
            track_boxes = np.array([])
        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)
        mot_accum.update(
            gt_ids,
            track_ids,
            distance)
        if i == len(seq)-1:
            break
    return mot_accum