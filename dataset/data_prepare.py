import os
import os.path as ops
import shutil

import argparse
import glob
import json
import cv2


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='The origin path of dataset')
    parser.add_argument('--output_dir', type=str, help='The output path of aug dataset in unauged form')
    return parser.parse_args()


def preprocess_aug_dataset(root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for pkt in glob.glob('{:s}/*calied'.format(root)):
        ann_path = ops.join(pkt, 'ann')
        img_path = ops.join(pkt, 'img')
        folder_name = ops.split(pkt)[1]
        out_folder_name = ops.join(output_dir, folder_name)
        os.makedirs(ops.join(out_folder_name, 'img'), exist_ok=True)
        os.makedirs(ops.join(out_folder_name, 'ann'), exist_ok=True)
        for idx, src_frame_name in enumerate(os.listdir(img_path)):
            outimg_name = 'frame_{:s}.png'.format('{:d}'.format(idx).zfill(5))
            outimg_path = ops.join(out_folder_name, 'img', outimg_name)

            outjson_name = outimg_name + '.json'
            outjson_path = ops.join(out_folder_name, 'ann', outjson_name)

            src_json_name = src_frame_name + '.json'
            src_frame_path = ops.join(img_path, src_frame_name)
            src_json_path = ops.join(ann_path, src_json_name)

            src_image = cv2.imread(src_frame_path, cv2.IMREAD_COLOR)
            cv2.imwrite(outimg_path, src_image)

            shutil.copyfile(src_json_path, outjson_path)
        print('Finish processing {}'.format(pkt))


if __name__ == '__main__':
    args = init_args()
    preprocess_aug_dataset(args.root, args.output_dir)