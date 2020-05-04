import os
import os.path as ops

from PIL import Image
import numpy as np
import torch

from torchvision.transforms import ToTensor

import csv


class COSMOSTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_track, det_root_txt):
        self.root_track = root_track
        self.det_root_txt = det_root_txt
        self.transforms = ToTensor()

        img_dir_path = ops.join(self.root_track, 'img1')
        gt_txt_path = ops.join(self.root_track, 'gt', 'gt.txt')

        self._img_paths = {}
        img_names = []
        for img_name in os.listdir(img_dir_path):
            img_names.append(img_name)
        img_names.sort()
        for i in range(len(img_names)):
            self._img_paths[i + 1] = ops.join(img_dir_path, img_names[i])

        self.anns_p = {i: {} for i in range(1, len(self._img_paths) + 1)}
        self.anns_v = {i: {} for i in range(1, len(self._img_paths) + 1)}
        self.vises_p = {i: {} for i in range(1, len(self._img_paths) + 1)}
        self.vises_v = {i: {} for i in range(1, len(self._img_paths) + 1)}
        with open(gt_txt_path, 'r') as file:
            data = file.readlines()
            for l in data:
                line = l.split(',')
                if line[-1].split()[0] == '2':
                    # pedestrians
                    self.anns_p[int(line[0])][int(line[1])] = np.array(
                        [int(line[2]), int(line[3]), int(line[2]) + int(line[4]), int(line[3]) + int(line[5])],
                        dtype='float32')
                    self.vises_p[int(line[0])][int(line[1])] = 1.0
                else:
                    # vehicle
                    self.anns_v[int(line[0])][int(line[1])] = np.array(
                        [int(line[2]), int(line[3]), int(line[2]) + int(line[4]), int(line[3]) + int(line[5])],
                        dtype='float32')
                    self.vises_v[int(line[0])][int(line[1])] = 1.0

        # detection result:
        self.det_p = {i: [] for i in range(1, len(self._img_paths) + 1)}
        self.det_v = {i: [] for i in range(1, len(self._img_paths) + 1)}
        self.score_p = {i: [] for i in range(1, len(self._img_paths) + 1)}
        self.score_v = {i: [] for i in range(1, len(self._img_paths) + 1)}
        with open(self.det_root_txt, 'r') as file:
            data = file.readlines()
            for l in data:
                line = l.split(',')
                if line[-2] == '2':
                    # pedestrians
                    self.det_p[int(line[0])].append(
                        [int(line[2]), int(line[3]), int(line[2]) + int(line[4]), int(line[3]) + int(line[5])])
                    self.score_p[int(line[0])].append(float(line[-1].split()[0]))
                else:
                    # vehicle
                    self.det_v[int(line[0])].append(
                        [int(line[2]), int(line[3]), int(line[2]) + int(line[4]), int(line[3]) + int(line[5])])
                    self.score_v[int(line[0])].append(float(line[-1].split()[0]))

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx + 1]
        img = Image.open(frame_img_path).convert("RGB")
        img = self.transforms(img)
        sample = {}
        sample['img'] = img

        sample['dets_p'] = torch.tensor(self.det_p[idx + 1], dtype=torch.float)
        sample['dets_v'] = torch.tensor(self.det_v[idx + 1], dtype=torch.float)
        sample['score_p'] = torch.tensor(self.score_p[idx + 1], dtype=torch.float)
        sample['score_v'] = torch.tensor(self.score_v[idx + 1], dtype=torch.float)

        sample['img_path'] = frame_img_path
        sample['gt_p'] = self.anns_p[idx + 1]
        sample['gt_v'] = self.anns_v[idx + 1]
        sample['vis_p'] = self.vises_p[idx + 1]
        sample['vis_v'] = self.vises_v[idx + 1]
        return sample

    def __len__(self):
        return len(self._img_paths)

    def __str__(self):
        return ops.split(self.root_track)[1]

    def write_results(self, results_p, results_v, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission
        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num
        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_p = ops.join(output_dir, ops.split(self.root_track)[1] + '_p.txt')
        file_v = ops.join(output_dir, ops.split(self.root_track)[1] + '_v.txt')
        with open(file_p, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in results_p.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])
        with open(file_v, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in results_v.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])





