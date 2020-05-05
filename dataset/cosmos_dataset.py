import configparser
import csv
import os
import os.path as ops
import pickle

from PIL import Image
import numpy as np
import scipy
import torch

import glob
import json
import cv2


class COSMOSDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self._classes = ('Background', 'Vehicle', 'Pedestrian')
        self._img_paths = {}
        img_num = 0
        for mp4file in glob.glob('{:s}/*.mp4'.format(self.root)):
            ann_path = ops.join(mp4file, 'ann')
            img_path = ops.join(mp4file, 'img')
            for img_name in os.listdir(img_path):
                frame_img_path = ops.join(img_path, img_name)
                frame_ann_name = ops.split(frame_img_path)[1] + '.json'
                frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
                num_objs = 0
                with open(frame_ann_path, 'r') as file:
                    info_dict = json.loads(file.readline())
                    for k in info_dict['objects']:
                        if k['classTitle'] == 'Vehicle' or k['classTitle'] == 'Pedestrian':
                            num_objs += 1
                if num_objs > 0:
                    self._img_paths[img_num] = ops.join(img_path, img_name)
                    img_num += 1

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        frame_img_path = self._img_paths[idx]
        frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
        with open(frame_ann_path, 'r') as file:
            info_dict = json.loads(file.readline())
            # frame_description = info_dict['description']
            # frame_tags = info_dict['tags']
            # frame_size = info_dict['size']
            # num_objs = len(info_dict['objects'])
            num_objs = 0
            for k in info_dict['objects']:
                if k['classTitle'] == 'Vehicle' or k['classTitle'] == 'Pedestrian':
                    num_objs += 1
            boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            num = 0
            for i, obj in enumerate(info_dict['objects']):
                if obj['classTitle'] == 'Pedestrian':
                    boxes[num, 0] = obj['points']['exterior'][0][0]
                    boxes[num, 1] = obj['points']['exterior'][0][1]
                    boxes[num, 2] = obj['points']['exterior'][1][0]
                    boxes[num, 3] = obj['points']['exterior'][1][1]
                    labels[num] = 2
                    num += 1
                elif obj['classTitle'] == 'Vehicle':
                    boxes[num, 0] = obj['points']['exterior'][0][0]
                    boxes[num, 1] = obj['points']['exterior'][0][1]
                    boxes[num, 2] = obj['points']['exterior'][1][0]
                    boxes[num, 3] = obj['points']['exterior'][1][1]
                    labels[num] = 1
                    num += 1
                else:
                    continue

        return {'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': iscrowd}

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx]
        # frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        # frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)

        img = Image.open(frame_img_path).convert("RGB")
        target = self._get_annotation(idx)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def print_eval(self, dataset_len, results, ovthresh=0.5, cls='Vehicle'):
        tp = [[] for _ in range(dataset_len)]
        fp = [[] for _ in range(dataset_len)]
        npos = 0
        gt = []
        gt_found = []
        im_indexes = []
        for idx, _ in results.items():
            annotation = self._get_annotation(idx)
            _bbox = annotation['boxes']
            lab = annotation['labels']
            if cls == 'Vehicle':
                bbox = _bbox[lab == 1, :]
            else:
                bbox = _bbox[lab == 2, :]
            found = np.zeros(bbox.shape[0])
            gt.append(bbox.cpu().numpy())
            gt_found.append(found)
            npos += found.shape[0]
            im_indexes.append(idx)
        # Loop through all images
        # for res in results:
        for j, (im_gt, found) in enumerate(zip(gt, gt_found)):
            im_index = im_indexes[j]
            if cls == 'Vehicle':
                im_det = results[im_index]['boxes'][results[im_index]['labels'] == 1, :].cpu().numpy()
            else:
                im_det = results[im_index]['boxes'][results[im_index]['labels'] == 2, :].cpu().numpy()
            im_tp = np.zeros(len(im_det))
            im_fp = np.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -np.inf
                if im_gt.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(im_gt[:, 0], d[0])
                    iymin = np.maximum(im_gt[:, 1], d[1])
                    ixmax = np.minimum(im_gt[:, 2], d[2])
                    iymax = np.minimum(im_gt[:, 3], d[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                           (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                           (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.
            tp[j] = im_tp
            fp[j] = im_fp
        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]
        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)
        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s + i] = tp_im
                fp_flat[i:s + i] = fp_im
                i += s
        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap
        if cls == 'Vehicle':
            print(f"Detection for Vehicle: AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")
        else:
            print(f"Detection for Pedestrian: AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")
        # print(f"AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")


class COSMOSDataset_AUG(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self._classes = ('Background', 'Vehicle', 'Pedestrian')
        self._img_paths = {}
        img_num = 0
        for pkt in glob.glob('{:s}/*calied'.format(self.root)):
            ann_path = ops.join(pkt, 'ann')
            img_path = ops.join(pkt, 'img')
            for img_name in os.listdir(img_path):
                frame_img_path = ops.join(img_path, img_name)
                frame_ann_name = ops.split(frame_img_path)[1] + '.json'
                frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
                num_objs = 0
                with open(frame_ann_path, 'r') as file:
                    info_dict = json.loads(file.readline())
                    for k in info_dict['objects']:
                        if k['classTitle'] == 'Vehicle' or k['classTitle'] == 'Pedestrian':
                            num_objs += 1
                if num_objs > 0:
                    self._img_paths[img_num] = ops.join(img_path, img_name)
                    img_num += 1

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        frame_img_path = self._img_paths[idx]
        frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
        with open(frame_ann_path, 'r') as file:
            info_dict = json.loads(file.readline())
            # frame_description = info_dict['description']
            # frame_tags = info_dict['tags']
            # frame_size = info_dict['size']
            # num_objs = len(info_dict['objects'])
            num_objs = 0
            for k in info_dict['objects']:
                if k['classTitle'] == 'Vehicle' or k['classTitle'] == 'Pedestrian':
                    num_objs += 1

            boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            num = 0

            for i, obj in enumerate(info_dict['objects']):
                if obj['classTitle'] == 'Pedestrian':
                    info_points = obj['points']['exterior']
                    x0, x1 = info_points[0]
                    y0, y1 = info_points[1]
                    for pt in info_points:
                        if pt[0] < x0:
                            x0 = pt[0]
                        if pt[1] < y0:
                            y0 = pt[1]
                        if pt[0] > x1:
                            x1 = pt[0]
                        if pt[1] > y1:
                            y1 = pt[1]
                    boxes[num, 0] = x0
                    boxes[num, 1] = y0
                    boxes[num, 2] = x1
                    boxes[num, 3] = y1
                    labels[num] = 2
                    num += 1
                elif obj['classTitle'] == 'Vehicle':
                    info_points = obj['points']['exterior']
                    x0, x1 = info_points[0]
                    y0, y1 = info_points[1]
                    for pt in info_points:
                        if pt[0] < x0:
                            x0 = pt[0]
                        if pt[1] < y0:
                            y0 = pt[1]
                        if pt[0] > x1:
                            x1 = pt[0]
                        if pt[1] > y1:
                            y1 = pt[1]
                    boxes[num, 0] = x0
                    boxes[num, 1] = y0
                    boxes[num, 2] = x1
                    boxes[num, 3] = y1
                    labels[num] = 1
                    num += 1
                else:
                    continue

        return {'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': iscrowd}

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx]
        # frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        # frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)

        img = Image.open(frame_img_path).convert("RGB")
        target = self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def print_eval(self, dataset_len, results, ovthresh=0.5, cls='Vehicle'):
        tp = [[] for _ in range(dataset_len)]
        fp = [[] for _ in range(dataset_len)]
        npos = 0
        gt = []
        gt_found = []
        im_indexes = []
        for idx, _ in results.items():
            annotation = self._get_annotation(idx)
            _bbox = annotation['boxes']
            lab = annotation['labels']
            if cls == 'Vehicle':
                bbox = _bbox[lab == 1, :]
            else:
                bbox = _bbox[lab == 2, :]
            found = np.zeros(bbox.shape[0])
            gt.append(bbox.cpu().numpy())
            gt_found.append(found)
            npos += found.shape[0]
            im_indexes.append(idx)
        # Loop through all images
        # for res in results:
        for j, (im_gt, found) in enumerate(zip(gt, gt_found)):
            im_index = im_indexes[j]
            if cls == 'Vehicle':
                im_det = results[im_index]['boxes'][results[im_index]['labels'] == 1, :].cpu().numpy()
            else:
                im_det = results[im_index]['boxes'][results[im_index]['labels'] == 2, :].cpu().numpy()
            im_tp = np.zeros(len(im_det))
            im_fp = np.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -np.inf
                if im_gt.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(im_gt[:, 0], d[0])
                    iymin = np.maximum(im_gt[:, 1], d[1])
                    ixmax = np.minimum(im_gt[:, 2], d[2])
                    iymax = np.minimum(im_gt[:, 3], d[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                           (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                           (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.
            tp[j] = im_tp
            fp[j] = im_fp
        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]
        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)
        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s + i] = tp_im
                fp_flat[i:s + i] = fp_im
                i += s
        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap
        if cls == 'Vehicle':
            print(f"Detection for Vehicle: AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")
        else:
            print(f"Detection for Pedestrian: AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")
        # print(f"AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")


class COSMOSDatasetTtoD(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.img1_path = ops.join(root, 'img1')
        self.gt_path = ops.join(root, 'gt', 'gt.txt')
        self.transforms = transforms
        self._classes = ('Background', 'Vehicle', 'Pedestrian')

        self.anns = {}
        self.anns_label = {}
        self._img_paths = {}
        for img_name in os.listdir(self.img1_path):
            frame_img_path = ops.join(self.img1_path, img_name)
            frame_num = int(img_name.split('.')[0])
            self._img_paths[frame_num] = frame_img_path
        for key in self._img_paths.keys():
            self.anns[key] = []
            self.anns_label[key] = []
        with open(self.gt_path, 'r') as file:
            data = file.readlines()
            for l in data:
                line = l.split(',')
                self.anns[int(line[0])].append(
                    [int(line[2]), int(line[3]), int(line[2]) + int(line[4]), int(line[3]) + int(line[5])])
                self.anns_label[int(line[0])].append(int(line[-1].split()[0]))

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        boxes = torch.tensor(self.anns[idx + 1], dtype=torch.float32)
        num_objs = boxes.size()[0]
        labels = torch.tensor(self.anns_label[idx + 1], dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        return {'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': iscrowd}

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx + 1]
        img = Image.open(frame_img_path).convert("RGB")
        target = self._get_annotation(idx)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def print_eval(self, dataset_len, results, ovthresh=0.5, cls='Vehicle'):
        tp = [[] for _ in range(dataset_len)]
        fp = [[] for _ in range(dataset_len)]
        npos = 0
        gt = []
        gt_found = []
        im_indexes = []
        for idx, _ in results.items():
            annotation = self._get_annotation(idx)
            _bbox = annotation['boxes']
            lab = annotation['labels']
            if cls == 'Vehicle':
                bbox = _bbox[lab == 1, :]
            else:
                bbox = _bbox[lab == 2, :]
            found = np.zeros(bbox.shape[0])
            gt.append(bbox.cpu().numpy())
            gt_found.append(found)
            npos += found.shape[0]
            im_indexes.append(idx)
        # Loop through all images
        # for res in results:
        for j, (im_gt, found) in enumerate(zip(gt, gt_found)):
            im_index = im_indexes[j]
            if cls == 'Vehicle':
                im_det = results[im_index]['boxes'][results[im_index]['labels'] == 1, :].cpu().numpy()
            else:
                im_det = results[im_index]['boxes'][results[im_index]['labels'] == 2, :].cpu().numpy()
            im_tp = np.zeros(len(im_det))
            im_fp = np.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -np.inf
                if im_gt.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(im_gt[:, 0], d[0])
                    iymin = np.maximum(im_gt[:, 1], d[1])
                    ixmax = np.minimum(im_gt[:, 2], d[2])
                    iymax = np.minimum(im_gt[:, 3], d[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                           (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                           (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.
            tp[j] = im_tp
            fp[j] = im_fp
        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]
        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)
        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s + i] = tp_im
                fp_flat[i:s + i] = fp_im
                i += s
        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap
        if cls == 'Vehicle':
            print(f"Detection for Vehicle: AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")
        else:
            print(f"Detection for Pedestrian: AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")
        # print(f"AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")


class COSMOSDatasetPedestrian(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self._classes = ('Background', 'Pedestrian')
        self._img_paths = {}
        img_num = 0
        for mp4file in glob.glob('{:s}/*.mp4'.format(self.root)):
            ann_path = ops.join(mp4file, 'ann')
            img_path = ops.join(mp4file, 'img')
            for img_name in os.listdir(img_path):
                frame_img_path = ops.join(img_path, img_name)
                frame_ann_name = ops.split(frame_img_path)[1] + '.json'
                frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
                num_objs = 0
                with open(frame_ann_path, 'r') as file:
                    info_dict = json.loads(file.readline())
                    for k in info_dict['objects']:
                        if k['classTitle'] == 'Pedestrian':
                            num_objs += 1
                if num_objs > 0:
                    self._img_paths[img_num] = ops.join(img_path, img_name)
                    img_num += 1

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        frame_img_path = self._img_paths[idx]
        frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
        with open(frame_ann_path, 'r') as file:
            info_dict = json.loads(file.readline())
            # frame_description = info_dict['description']
            # frame_tags = info_dict['tags']
            # frame_size = info_dict['size']
            # num_objs = len(info_dict['objects'])
            num_objs = 0
            for k in info_dict['objects']:
                if k['classTitle'] == 'Pedestrian':
                    num_objs += 1
            boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            num = 0
            for i, obj in enumerate(info_dict['objects']):
                if obj['classTitle'] == 'Vehicle':
                    pass
                elif obj['classTitle'] == 'Pedestrian':
                    boxes[num, 0] = obj['points']['exterior'][0][0]
                    boxes[num, 1] = obj['points']['exterior'][0][1]
                    boxes[num, 2] = obj['points']['exterior'][1][0]
                    boxes[num, 3] = obj['points']['exterior'][1][1]
                    num += 1
                else:
                    labels[num] = 0
                    num += 1

        return {'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': iscrowd}

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx]
        # frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        # frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)

        img = Image.open(frame_img_path).convert("RGB")
        target = self._get_annotation(idx)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def write_results_files(self, results, output_dir):
        files = {}
        for image_id, res in results.items():
            path = self._img_paths[image_id]
            img_folder, frame_name = ops.split(path)
            frame = int(frame_name.split('.')[0].split('_')[1])
            # get image number out of name
            out = img_folder.split('/')[-2] + '.txt'
            outfile = ops.join(output_dir, out)

            if outfile not in files.keys():
                files[outfile] = []
            for box, label, score in zip(res['boxes'], res['labels'], res['scores']):
                x1 = int(box[0].item())
                y1 = int(box[1].item())
                x2 = int(box[2].item())
                y2 = int(box[3].item())
                files[outfile].append([frame, label.item(), x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])
        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)

    def print_eval(self, dataset_len, results, ovthresh=0.5):
        # Lists for tp and fp in the format tp[cls][image]
        tp = [[] for _ in range(dataset_len)]
        fp = [[] for _ in range(dataset_len)]

        npos = 0
        gt = []
        gt_found = []
        im_indexes = []
        for idx, _ in results.items():
            annotation = self._get_annotation(idx)
            bbox = annotation['boxes']
            found = np.zeros(bbox.shape[0])
            gt.append(bbox.cpu().numpy())
            gt_found.append(found)
            npos += found.shape[0]
            im_indexes.append(idx)

        # Loop through all images
        # for res in results:
        for j, (im_gt, found) in enumerate(zip(gt, gt_found)):
            im_index = im_indexes[j]
            im_det = results[im_index]['boxes'].cpu().numpy()

            im_tp = np.zeros(len(im_det))
            im_fp = np.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -np.inf
                if im_gt.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(im_gt[:, 0], d[0])
                    iymin = np.maximum(im_gt[:, 1], d[1])
                    ixmax = np.minimum(im_gt[:, 2], d[2])
                    iymax = np.minimum(im_gt[:, 3], d[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                           (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                           (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.
            tp[j] = im_tp
            fp[j] = im_fp
        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]
        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)
        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s + i] = tp_im
                fp_flat[i:s + i] = fp_im
                i += s
        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap

        print(f"AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")


class COSMOSDatasetPedestrian_AUG(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self._classes = ('Background', 'Pedestrian')
        self._img_paths = {}
        img_num = 0
        for pkt in glob.glob('{:s}/*calied'.format(self.root)):
            ann_path = ops.join(pkt, 'ann')
            img_path = ops.join(pkt, 'img')
            for img_name in os.listdir(img_path):
                frame_img_path = ops.join(img_path, img_name)
                frame_ann_name = ops.split(frame_img_path)[1] + '.json'
                frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
                num_objs = 0
                with open(frame_ann_path, 'r') as file:
                    info_dict = json.loads(file.readline())
                    for k in info_dict['objects']:
                        if k['classTitle'] == 'Pedestrian':
                            num_objs += 1
                if num_objs > 0:
                    self._img_paths[img_num] = ops.join(img_path, img_name)
                    img_num += 1

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        frame_img_path = self._img_paths[idx]
        frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
        with open(frame_ann_path, 'r') as file:
            info_dict = json.loads(file.readline())
            # frame_description = info_dict['description']
            # frame_tags = info_dict['tags']
            # frame_size = info_dict['size']
            # num_objs = len(info_dict['objects'])
            num_objs = 0
            for k in info_dict['objects']:
                if k['classTitle'] == 'Pedestrian':
                    num_objs += 1

            boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            num = 0

            for i, obj in enumerate(info_dict['objects']):
                if obj['classTitle'] == 'Vehicle':
                    pass
                elif obj['classTitle'] == 'Pedestrian':
                    info_points = obj['points']['exterior']
                    x0, x1 = info_points[0]
                    y0, y1 = info_points[1]
                    for pt in info_points:
                        if pt[0] < x0:
                            x0 = pt[0]
                        if pt[1] < y0:
                            y0 = pt[1]
                        if pt[0] > x1:
                            x1 = pt[0]
                        if pt[1] > y1:
                            y1 = pt[1]
                    boxes[num, 0] = x0
                    boxes[num, 1] = y0
                    boxes[num, 2] = x1
                    boxes[num, 3] = y1
                    num += 1
                else:
                    pass

        return {'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': iscrowd}

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx]
        # frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        # frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)

        img = Image.open(frame_img_path).convert("RGB")
        target = self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)

    def write_results_files(self, results, output_dir):
        files = {}
        for image_id, res in results.items():
            path = self._img_paths[image_id]
            img_folder, frame_name = ops.split(path)
            frame = int(frame_name.split('.')[0].split('_')[1])
            # get image number out of name
            out = img_folder.split('/')[-2] + '.txt'
            outfile = ops.join(output_dir, out)

            if outfile not in files.keys():
                files[outfile] = []
            for box, label, score in zip(res['boxes'], res['labels'], res['scores']):
                x1 = int(box[0].item())
                y1 = int(box[1].item())
                x2 = int(box[2].item())
                y2 = int(box[3].item())
                files[outfile].append([frame, label.item(), x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])
        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)

    def print_eval(self, dataset_len, results, ovthresh=0.5):
        # Lists for tp and fp in the format tp[cls][image]
        tp = [[] for _ in range(dataset_len)]
        fp = [[] for _ in range(dataset_len)]

        npos = 0
        gt = []
        gt_found = []
        im_indexes = []
        for idx, _ in results.items():
            annotation = self._get_annotation(idx)
            bbox = annotation['boxes']
            found = np.zeros(bbox.shape[0])
            gt.append(bbox.cpu().numpy())
            gt_found.append(found)
            npos += found.shape[0]
            im_indexes.append(idx)

        # Loop through all images
        # for res in results:
        for j, (im_gt, found) in enumerate(zip(gt, gt_found)):
            im_index = im_indexes[j]
            im_det = results[im_index]['boxes'].cpu().numpy()

            im_tp = np.zeros(len(im_det))
            im_fp = np.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -np.inf
                if im_gt.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(im_gt[:, 0], d[0])
                    iymin = np.maximum(im_gt[:, 1], d[1])
                    ixmax = np.minimum(im_gt[:, 2], d[2])
                    iymax = np.minimum(im_gt[:, 3], d[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                           (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                           (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.
            tp[j] = im_tp
            fp[j] = im_fp
        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]
        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)
        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s + i] = tp_im
                fp_flat[i:s + i] = fp_im
                i += s
        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap

        print(f"AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")

class COSMOSDatasetPedestrianTtoD(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.img1_path = ops.join(root, 'img1')
        self.gt_path = ops.join(root, 'gt', 'gt.txt')
        self.transforms = transforms
        self._classes = ('Background', 'Pedestrian')

        self.anns = {}
        self._img_paths = {}
        for img_name in os.listdir(self.img1_path):
            frame_img_path = ops.join(self.img1_path, img_name)
            frame_num = int(img_name.split('.')[0])
            self._img_paths[frame_num] = frame_img_path
        for key in self._img_paths.keys():
            self.anns[key] = []
        with open(self.gt_path, 'r') as file:
            data = file.readlines()
            for l in data:
                line = l.split(',')
                self.anns[int(line[0])].append(
                    [int(line[2]), int(line[3]), int(line[2]) + int(line[4]), int(line[3]) + int(line[5])])

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        boxes = torch.tensor(self.anns[idx + 1], dtype=torch.float32)
        num_objs = boxes.size()[0]
        labels = torch.ones((num_objs,), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        return {'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': iscrowd}

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx + 1]
        img = Image.open(frame_img_path).convert("RGB")
        target = self._get_annotation(idx)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def write_results_files(self, results, output_dir):
        files = {}
        for image_id, res in results.items():
            path = self._img_paths[image_id]
            img_folder, frame_name = ops.split(path)
            frame = int(frame_name.split('.')[0].split('_')[1])
            # get image number out of name
            out = img_folder.split('/')[-2] + '.txt'
            outfile = ops.join(output_dir, out)

            if outfile not in files.keys():
                files[outfile] = []
            for box, label, score in zip(res['boxes'], res['labels'], res['scores']):
                x1 = int(box[0].item())
                y1 = int(box[1].item())
                x2 = int(box[2].item())
                y2 = int(box[3].item())
                files[outfile].append([frame, label.item(), x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])
        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)

    def print_eval(self, dataset_len, results, ovthresh=0.5):
        # Lists for tp and fp in the format tp[cls][image]
        tp = [[] for _ in range(dataset_len)]
        fp = [[] for _ in range(dataset_len)]

        npos = 0
        gt = []
        gt_found = []
        im_indexes = []
        for idx, _ in results.items():
            annotation = self._get_annotation(idx)
            bbox = annotation['boxes']
            found = np.zeros(bbox.shape[0])
            gt.append(bbox.cpu().numpy())
            gt_found.append(found)
            npos += found.shape[0]
            im_indexes.append(idx)

        # Loop through all images
        # for res in results:
        for j, (im_gt, found) in enumerate(zip(gt, gt_found)):
            im_index = im_indexes[j]
            im_det = results[im_index]['boxes'].cpu().numpy()

            im_tp = np.zeros(len(im_det))
            im_fp = np.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -np.inf
                if im_gt.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(im_gt[:, 0], d[0])
                    iymin = np.maximum(im_gt[:, 1], d[1])
                    ixmax = np.minimum(im_gt[:, 2], d[2])
                    iymax = np.minimum(im_gt[:, 3], d[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                           (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                           (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.
            tp[j] = im_tp
            fp[j] = im_fp
        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]
        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)
        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s + i] = tp_im
                fp_flat[i:s + i] = fp_im
                i += s
        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap

        print(f"AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")

class COSMOSDatasetVehicle(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self._classes = ('Background', 'Vehicle')
        self._img_paths = {}
        img_num = 0
        for mp4file in glob.glob('{:s}/*.mp4'.format(self.root)):
            ann_path = ops.join(mp4file, 'ann')
            img_path = ops.join(mp4file, 'img')
            for img_name in os.listdir(img_path):
                frame_img_path = ops.join(img_path, img_name)
                frame_ann_name = ops.split(frame_img_path)[1] + '.json'
                frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
                num_objs = 0
                with open(frame_ann_path, 'r') as file:
                    info_dict = json.loads(file.readline())
                    for k in info_dict['objects']:
                        if k['classTitle'] == 'Vehicle':
                            num_objs += 1
                if num_objs > 0:
                    self._img_paths[img_num] = ops.join(img_path, img_name)
                    img_num += 1

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        frame_img_path = self._img_paths[idx]
        frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
        with open(frame_ann_path, 'r') as file:
            info_dict = json.loads(file.readline())
            # frame_description = info_dict['description']
            # frame_tags = info_dict['tags']
            # frame_size = info_dict['size']
            # num_objs = len(info_dict['objects'])
            num_objs = 0
            for k in info_dict['objects']:
                if k['classTitle'] == 'Vehicle':
                    num_objs += 1
            boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            num = 0
            for i, obj in enumerate(info_dict['objects']):
                if obj['classTitle'] == 'Pedestrian':
                    pass
                elif obj['classTitle'] == 'Vehicle':
                    boxes[num, 0] = obj['points']['exterior'][0][0]
                    boxes[num, 1] = obj['points']['exterior'][0][1]
                    boxes[num, 2] = obj['points']['exterior'][1][0]
                    boxes[num, 3] = obj['points']['exterior'][1][1]
                    num += 1
                else:
                    labels[num] = 0
                    num += 1

        return {'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': iscrowd}

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx]
        # frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        # frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)

        img = Image.open(frame_img_path).convert("RGB")
        target = self._get_annotation(idx)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def write_results_files(self, results, output_dir):
        files = {}
        for image_id, res in results.items():
            path = self._img_paths[image_id]
            img_folder, frame_name = ops.split(path)
            frame = int(frame_name.split('.')[0].split('_')[1])
            # get image number out of name
            out = img_folder.split('/')[-2] + '.txt'
            outfile = ops.join(output_dir, out)

            if outfile not in files.keys():
                files[outfile] = []
            for box, label, score in zip(res['boxes'], res['labels'], res['scores']):
                x1 = int(box[0].item())
                y1 = int(box[1].item())
                x2 = int(box[2].item())
                y2 = int(box[3].item())
                files[outfile].append([frame, label.item(), x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])
        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)

    def print_eval(self, dataset_len, results, ovthresh=0.5):
        # Lists for tp and fp in the format tp[cls][image]
        tp = [[] for _ in range(dataset_len)]
        fp = [[] for _ in range(dataset_len)]

        npos = 0
        gt = []
        gt_found = []
        im_indexes = []
        for idx, _ in results.items():
            annotation = self._get_annotation(idx)
            bbox = annotation['boxes']
            found = np.zeros(bbox.shape[0])
            gt.append(bbox.cpu().numpy())
            gt_found.append(found)
            npos += found.shape[0]
            im_indexes.append(idx)

        # Loop through all images
        # for res in results:
        for j, (im_gt, found) in enumerate(zip(gt, gt_found)):
            im_index = im_indexes[j]
            im_det = results[im_index]['boxes'].cpu().numpy()

            im_tp = np.zeros(len(im_det))
            im_fp = np.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -np.inf
                if im_gt.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(im_gt[:, 0], d[0])
                    iymin = np.maximum(im_gt[:, 1], d[1])
                    ixmax = np.minimum(im_gt[:, 2], d[2])
                    iymax = np.minimum(im_gt[:, 3], d[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                           (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                           (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.
            tp[j] = im_tp
            fp[j] = im_fp
        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]
        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)
        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s + i] = tp_im
                fp_flat[i:s + i] = fp_im
                i += s
        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap

        print(f"AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")


class COSMOSDatasetVehicle_AUG(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self._classes = ('Background', 'Vehicle')
        self._img_paths = {}
        img_num = 0
        for pkt in glob.glob('{:s}/*calied'.format(self.root)):
            ann_path = ops.join(pkt, 'ann')
            img_path = ops.join(pkt, 'img')
            for img_name in os.listdir(img_path):
                frame_img_path = ops.join(img_path, img_name)
                frame_ann_name = ops.split(frame_img_path)[1] + '.json'
                frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
                num_objs = 0
                with open(frame_ann_path, 'r') as file:
                    info_dict = json.loads(file.readline())
                    for k in info_dict['objects']:
                        if k['classTitle'] == 'Vehicle':
                            num_objs += 1
                if num_objs > 0:
                    self._img_paths[img_num] = ops.join(img_path, img_name)
                    img_num += 1

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        frame_img_path = self._img_paths[idx]
        frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)
        with open(frame_ann_path, 'r') as file:
            info_dict = json.loads(file.readline())
            # frame_description = info_dict['description']
            # frame_tags = info_dict['tags']
            # frame_size = info_dict['size']
            # num_objs = len(info_dict['objects'])
            num_objs = 0
            for k in info_dict['objects']:
                if k['classTitle'] == 'Vehicle':
                    num_objs += 1

            boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            num = 0

            for i, obj in enumerate(info_dict['objects']):
                if obj['classTitle'] == 'Pedestrian':
                    pass
                elif obj['classTitle'] == 'Vehicle':
                    info_points = obj['points']['exterior']
                    x0, x1 = info_points[0]
                    y0, y1 = info_points[1]
                    for pt in info_points:
                        if pt[0] < x0:
                            x0 = pt[0]
                        if pt[1] < y0:
                            y0 = pt[1]
                        if pt[0] > x1:
                            x1 = pt[0]
                        if pt[1] > y1:
                            y1 = pt[1]
                    boxes[num, 0] = x0
                    boxes[num, 1] = y0
                    boxes[num, 2] = x1
                    boxes[num, 3] = y1
                    num += 1
                else:
                    pass

        return {'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': iscrowd}

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx]
        # frame_ann_name = ops.split(frame_img_path)[1] + '.json'
        # frame_ann_path = ops.join(ops.split(ops.split(frame_img_path)[0])[0], 'ann/' + frame_ann_name)

        img = Image.open(frame_img_path).convert("RGB")
        target = self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)

    def write_results_files(self, results, output_dir):
        files = {}
        for image_id, res in results.items():
            path = self._img_paths[image_id]
            img_folder, frame_name = ops.split(path)
            frame = int(frame_name.split('.')[0].split('_')[1])
            # get image number out of name
            out = img_folder.split('/')[-2] + '.txt'
            outfile = ops.join(output_dir, out)

            if outfile not in files.keys():
                files[outfile] = []
            for box, label, score in zip(res['boxes'], res['labels'], res['scores']):
                x1 = int(box[0].item())
                y1 = int(box[1].item())
                x2 = int(box[2].item())
                y2 = int(box[3].item())
                files[outfile].append([frame, label.item(), x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])
        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)

    def print_eval(self, dataset_len, results, ovthresh=0.5):
        # Lists for tp and fp in the format tp[cls][image]
        tp = [[] for _ in range(dataset_len)]
        fp = [[] for _ in range(dataset_len)]

        npos = 0
        gt = []
        gt_found = []
        im_indexes = []
        for idx, _ in results.items():
            annotation = self._get_annotation(idx)
            bbox = annotation['boxes']
            found = np.zeros(bbox.shape[0])
            gt.append(bbox.cpu().numpy())
            gt_found.append(found)
            npos += found.shape[0]
            im_indexes.append(idx)

        # Loop through all images
        # for res in results:
        for j, (im_gt, found) in enumerate(zip(gt, gt_found)):
            im_index = im_indexes[j]
            im_det = results[im_index]['boxes'].cpu().numpy()

            im_tp = np.zeros(len(im_det))
            im_fp = np.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -np.inf
                if im_gt.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(im_gt[:, 0], d[0])
                    iymin = np.maximum(im_gt[:, 1], d[1])
                    ixmax = np.minimum(im_gt[:, 2], d[2])
                    iymax = np.minimum(im_gt[:, 3], d[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                           (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                           (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.
            tp[j] = im_tp
            fp[j] = im_fp
        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]
        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)
        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s + i] = tp_im
                fp_flat[i:s + i] = fp_im
                i += s
        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap

        print(f"AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")


class COSMOSDatasetVehicleTtoD(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.img1_path = ops.join(root, 'img1')
        self.gt_path = ops.join(root, 'gt', 'gt.txt')
        self.transforms = transforms
        self._classes = ('Background', 'Vehicle')

        self.anns = {}
        self._img_paths = {}
        for img_name in os.listdir(self.img1_path):
            frame_img_path = ops.join(self.img1_path, img_name)
            frame_num = int(img_name.split('.')[0])
            self._img_paths[frame_num] = frame_img_path
        for key in self._img_paths.keys():
            self.anns[key] = []
        with open(self.gt_path, 'r') as file:
            data = file.readlines()
            for l in data:
                line = l.split(',')
                self.anns[int(line[0])].append(
                    [int(line[2]), int(line[3]), int(line[2]) + int(line[4]), int(line[3]) + int(line[5])])

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        boxes = torch.tensor(self.anns[idx + 1], dtype=torch.float32)
        num_objs = boxes.size()[0]
        labels = torch.ones((num_objs,), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        return {'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': iscrowd}

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        frame_img_path = self._img_paths[idx + 1]
        img = Image.open(frame_img_path).convert("RGB")
        target = self._get_annotation(idx)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def write_results_files(self, results, output_dir):
        files = {}
        for image_id, res in results.items():
            path = self._img_paths[image_id]
            img_folder, frame_name = ops.split(path)
            frame = int(frame_name.split('.')[0].split('_')[1])
            # get image number out of name
            out = img_folder.split('/')[-2] + '.txt'
            outfile = ops.join(output_dir, out)

            if outfile not in files.keys():
                files[outfile] = []
            for box, label, score in zip(res['boxes'], res['labels'], res['scores']):
                x1 = int(box[0].item())
                y1 = int(box[1].item())
                x2 = int(box[2].item())
                y2 = int(box[3].item())
                files[outfile].append([frame, label.item(), x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])
        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)

    def print_eval(self, dataset_len, results, ovthresh=0.5):
        # Lists for tp and fp in the format tp[cls][image]
        tp = [[] for _ in range(dataset_len)]
        fp = [[] for _ in range(dataset_len)]

        npos = 0
        gt = []
        gt_found = []
        im_indexes = []
        for idx, _ in results.items():
            annotation = self._get_annotation(idx)
            bbox = annotation['boxes']
            found = np.zeros(bbox.shape[0])
            gt.append(bbox.cpu().numpy())
            gt_found.append(found)
            npos += found.shape[0]
            im_indexes.append(idx)

        # Loop through all images
        # for res in results:
        for j, (im_gt, found) in enumerate(zip(gt, gt_found)):
            im_index = im_indexes[j]
            im_det = results[im_index]['boxes'].cpu().numpy()

            im_tp = np.zeros(len(im_det))
            im_fp = np.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -np.inf
                if im_gt.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(im_gt[:, 0], d[0])
                    iymin = np.maximum(im_gt[:, 1], d[1])
                    ixmax = np.minimum(im_gt[:, 2], d[2])
                    iymax = np.minimum(im_gt[:, 3], d[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                           (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                           (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.
            tp[j] = im_tp
            fp[j] = im_fp
        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]
        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)
        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s + i] = tp_im
                fp_flat[i:s + i] = fp_im
                i += s
        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        tp, fp, prec, rec, ap = np.max(tp), np.max(fp), prec[-1], np.max(rec), ap

        print(f"AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")


def preprocessvideo(video_name, json_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    outimg_dir = ops.join(output_dir, 'img1')
    outtxt_dir = ops.join(output_dir, 'gt')
    if not ops.exists(outimg_dir):
        os.makedirs(outimg_dir)
    if not ops.exists(outtxt_dir):
        os.makedirs(outtxt_dir)

    # process the image
    video_cap = cv2.VideoCapture(video_name)
    frame_count = 0
    all_frames = []
    while (True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1
    for i, frame in enumerate(all_frames):
        out_frame_name = '{:s}.png'.format('{:d}'.format(i + 1).zfill(6))
        out_frame_path = ops.join(outimg_dir, out_frame_name)
        cv2.imwrite(out_frame_path, frame)

    # process the ground truth
    idx = 1
    with open(json_name) as json_file:
        file = []
        data = json.load(json_file)
        for i, obj in enumerate(data["annotations"]["track"]):
            obj_label = obj['_label']
            if obj_label != "vehicle" and obj_label != "pedestrian":
                continue
            for info in obj['box']:
                cur_frame = str(int(info['_frame']) + 1)
                bb_left = int(float(info['_xtl']))
                bb_top = int(float(info['_ytl']))
                bb_width = int(float(info['_xbr'])) - int(float(info['_xtl']))
                bb_height = int(float(info['_ybr'])) - int(float(info['_ytl']))
                if obj_label == "vehicle":
                    file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1, 1])
                else:
                    file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1, 2])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def process_mask_to_txt(root, output_dir):
    # <ferame> <1> <left_top_x> <left_top_y> <x_range> <y_range> <label> <scores>
    file = []
    frame_num = len(os.listdir(root))
    for i in range(frame_num):
        detect_pkl_path = ops.join(root, '{:s}.pkl'.format('{:d}'.format(i).zfill(7)))
        fr = open(detect_pkl_path, 'rb')
        inf = pickle.load(fr)

        class_ids = inf['class_ids']
        scores = inf['scores']
        contours = inf['contours']
        num_obj = len(contours)
        for j in range(num_obj):
            label = class_ids[j]
            points = contours[j][0]
            if label == 1:
                x_left = y_left = np.inf
                x_right = y_right = 0
                for point in points:
                    x, y = point[0]
                    if x < x_left:
                        x_left = x
                    if y < y_left:
                        y_left = y
                    if x > x_right:
                        x_right = x
                    if y > y_right:
                        y_right = y
                file.append([i + 1, 1, x_left, y_left, x_right - x_left, y_right - y_left, 2, scores[j]])
            elif label == 2:
                x_left = y_left = np.inf
                x_right = y_right = 0
                for point in points:
                    x, y = point[0]
                    if x < x_left:
                        x_left = x
                    if y < y_left:
                        y_left = y
                    if x > x_right:
                        x_right = x
                    if y > y_right:
                        y_right = y
                file.append([i + 1, 1, x_left, y_left, x_right - x_left, y_right - y_left, 1, scores[j]])
            else:
                pass
    with open(ops.join(output_dir, '{}.txt'.format(ops.split(root)[1])), "w") as of:
        writer = csv.writer(of, delimiter=',')
        for d in file:
            writer.writerow(d)
    print('process {} finish'.format(ops.split(root)[1]))


def preprocessvideo_2(video_name, json_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    outimg_dir = ops.join(output_dir, 'img1')
    outtxt_dir = ops.join(output_dir, 'gt')
    if not ops.exists(outimg_dir):
        os.makedirs(outimg_dir)
    if not ops.exists(outtxt_dir):
        os.makedirs(outtxt_dir)

    # process the image
    video_cap = cv2.VideoCapture(video_name)
    frame_count = 0
    all_frames = []
    while (True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1
    for i, frame in enumerate(all_frames):
        out_frame_name = '{:s}.png'.format('{:d}'.format(i+1).zfill(6))
        out_frame_path = ops.join(outimg_dir, out_frame_name)
        cv2.imwrite(out_frame_path, frame)

    # process the ground truth
    idx = 1
    with open(json_name) as json_file:
        file = []
        data = json.load(json_file)
        for i, obj in enumerate(data["annotations"]["track"]):
            obj_label = obj['_label']
            if obj_label != "Car" and obj_label != "Pedestrian":
                continue
            for info in obj['box']:
                cur_frame = str(int(info['_frame']) + 1)
                bb_left = int(float(info['_xtl']))
                bb_top = int(float(info['_ytl']))
                bb_width = int(float(info['_xbr'])) - int(float(info['_xtl']))
                bb_height = int(float(info['_ybr'])) - int(float(info['_ytl']))
                if obj_label == "Car":
                    file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1, 1])
                else:
                    file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1, 2])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def preprocessvideo_3(video_name, json_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    outimg_dir = ops.join(output_dir, 'img1')
    outtxt_dir = ops.join(output_dir, 'gt')
    if not ops.exists(outimg_dir):
        os.makedirs(outimg_dir)
    if not ops.exists(outtxt_dir):
        os.makedirs(outtxt_dir)

    # process the image
    video_cap = cv2.VideoCapture(video_name)
    frame_count = 0
    all_frames = []
    while (True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1
    for i, frame in enumerate(all_frames):
        out_frame_name = '{:s}.png'.format('{:d}'.format(i+1).zfill(6))
        out_frame_path = ops.join(outimg_dir, out_frame_name)
        cv2.imwrite(out_frame_path, frame)

    # process the ground truth
    idx = 1
    with open(json_name) as json_file:
        file = []
        data = json.load(json_file)
        for i, obj in enumerate(data["annotations"]["track"]):
            obj_label = obj['_label']
            if obj_label != "Bus" and obj_label != "Car" and obj_label != "Pedestrian" and obj_label != "Truck":
                continue
            for info in obj['box']:
                cur_frame = str(int(info['_frame']) + 1)
                bb_left = int(float(info['_xtl']))
                bb_top = int(float(info['_ytl']))
                bb_width = int(float(info['_xbr'])) - int(float(info['_xtl']))
                bb_height = int(float(info['_ybr'])) - int(float(info['_ytl']))
                if obj_label == "Bus" or obj_label == "Car" or obj_label == "Truck":
                    file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1, 1])
                else:
                    file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1, 2])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def preprocessvideo_4(video_name, json_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    outimg_dir = ops.join(output_dir, 'img1')
    outtxt_dir = ops.join(output_dir, 'gt')
    if not ops.exists(outimg_dir):
        os.makedirs(outimg_dir)
    if not ops.exists(outtxt_dir):
        os.makedirs(outtxt_dir)

    # process the image
    video_cap = cv2.VideoCapture(video_name)
    frame_count = 0
    all_frames = []
    while (True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1
    for i, frame in enumerate(all_frames):
        out_frame_name = '{:s}.png'.format('{:d}'.format(i+1).zfill(6))
        out_frame_path = ops.join(outimg_dir, out_frame_name)
        cv2.imwrite(out_frame_path, frame)

    # process the ground truth
    idx = 1
    with open(json_name) as json_file:
        file = []
        data = json.load(json_file)
        for i, obj in enumerate(data["annotations"]["track"]):
            obj_label = obj['-label']
            if obj_label != "vehicle" and obj_label != "pedestrian":
                continue
            for info in obj['box']:
                cur_frame = str(int(info['-frame']) + 1)
                bb_left = int(float(info['-xtl']))
                bb_top = int(float(info['-ytl']))
                bb_width = int(float(info['-xbr'])) - int(float(info['-xtl']))
                bb_height = int(float(info['-ybr'])) - int(float(info['-ytl']))
                if obj_label == "vehicle":
                    file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1, 1])
                else:
                    file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1, 2])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def preprocessvideo_5(video_name, json_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    outimg_dir = ops.join(output_dir, 'img1')
    outtxt_dir = ops.join(output_dir, 'gt')
    if not ops.exists(outimg_dir):
        os.makedirs(outimg_dir)
    if not ops.exists(outtxt_dir):
        os.makedirs(outtxt_dir)

    # process the image
    video_cap = cv2.VideoCapture(video_name)
    frame_count = 0
    all_frames = []
    while (True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1
    for i, frame in enumerate(all_frames):
        out_frame_name = '{:s}.png'.format('{:d}'.format(i+1).zfill(6))
        out_frame_path = ops.join(outimg_dir, out_frame_name)
        cv2.imwrite(out_frame_path, frame)

    # process the ground truth
    idx = 1
    with open(json_name) as json_file:
        file = []
        data = json.load(json_file)
        for i, obj in enumerate(data["annotations"]["track"]):
            obj_label = obj['_label']
            if obj_label != "Vehicle" and obj_label != "Pedestrian":
                continue
            for info in obj['box']:
                cur_frame = str(int(info['_frame']) + 1)
                bb_left = int(float(info['_xtl']))
                bb_top = int(float(info['_ytl']))
                bb_width = int(float(info['_xbr'])) - int(float(info['_xtl']))
                bb_height = int(float(info['_ybr'])) - int(float(info['_ytl']))
                if obj_label == "Vehicle":
                    file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1, 1])
                else:
                    file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1, 2])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames
