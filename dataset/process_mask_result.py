import csv
import os
import os.path as ops
import pickle
import numpy as np
import argparse


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='The origin path of dataset')
    parser.add_argument('--output_dir', type=str, help='The output path for mask rcnn result')
    return parser.parse_args()


def process_mask_to_txt(root, output_dir):
    # <ferame> <1> <left_top_x> <left_top_y> <x_range> <y_range> <label> <scores>
    os.makedirs(output_dir, exist_ok=True)
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


if __name__ == '__main__':
    args = init_args()
    for mask_dir in os.listdir(args.root):
        if 'Mask' not in mask_dir:
            continue
        else:
            result_pkl_path = ops.join(args.root, mask_dir, 'detection_output', mask_dir.split('MaskRCNN_')[1]+'.mp4')
            detection_output_path = ops.join(args.root, mask_dir, 'detection_output')
            for video_name in os.listdir(detection_output_path):
                if '.mp4' in video_name:
                    result_pkl_path = ops.join(detection_output_path, video_name)
                    process_mask_to_txt(result_pkl_path, args.output_dir)
                else:
                    continue
    print('-----FINISH-----')