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


def preprocessvideo_p(video_name, json_name, output_dir, cls):
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
            if obj_label != cls:
                continue
            for info in obj['box']:
                cur_frame = str(int(info['_frame']) + 1)
                bb_left = int(float(info['_xtl']))
                bb_top = int(float(info['_ytl']))
                bb_width = int(float(info['_xbr'])) - int(float(info['_xtl']))
                bb_height = int(float(info['_ybr'])) - int(float(info['_ytl']))
                file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def preprocessvideo_p_2(video_name, json_name, output_dir, cls):
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
            if obj_label not in cls:
                continue
            for info in obj['box']:
                cur_frame = str(int(info['_frame']) + 1)
                bb_left = int(float(info['_xtl']))
                bb_top = int(float(info['_ytl']))
                bb_width = int(float(info['_xbr'])) - int(float(info['_xtl']))
                bb_height = int(float(info['_ybr'])) - int(float(info['_ytl']))
                file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def preprocessvideo_p_3(video_name, json_name, output_dir, cls):
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
            if obj_label not in cls:
                continue
            for info in obj['box']:
                cur_frame = str(int(info['-frame']) + 1)
                bb_left = int(float(info['-xtl']))
                bb_top = int(float(info['-ytl']))
                bb_width = int(float(info['-xbr'])) - int(float(info['-xtl']))
                bb_height = int(float(info['-ybr'])) - int(float(info['-ytl']))
                file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def preprocessvideo_p_4(video_name, json_name, output_dir, cls):
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
            if obj_label not in cls:
                continue
            for info in obj['box']:
                cur_frame = str(int(info['_frame']) + 1)
                bb_left = int(float(info['_xtl']))
                bb_top = int(float(info['_ytl']))
                bb_width = int(float(info['_xbr'])) - int(float(info['_xtl']))
                bb_height = int(float(info['_ybr'])) - int(float(info['_ytl']))
                file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def preprocessvideo_v(video_name, json_name, output_dir, cls):
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
            if obj_label != cls:
                continue
            for info in obj['box']:
                cur_frame = str(int(info['_frame']) + 1)
                bb_left = int(float(info['_xtl']))
                bb_top = int(float(info['_ytl']))
                bb_width = int(float(info['_xbr'])) - int(float(info['_xtl']))
                bb_height = int(float(info['_ybr'])) - int(float(info['_ytl']))
                file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def preprocessvideo_v_2(video_name, json_name, output_dir, cls):
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
            if obj_label not in cls:
                continue
            for info in obj['box']:
                cur_frame = str(int(info['_frame']) + 1)
                bb_left = int(float(info['_xtl']))
                bb_top = int(float(info['_ytl']))
                bb_width = int(float(info['_xbr'])) - int(float(info['_xtl']))
                bb_height = int(float(info['_ybr'])) - int(float(info['_ytl']))
                file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def preprocessvideo_v_3(video_name, json_name, output_dir, cls):
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
            if obj_label not in cls:
                continue
            for info in obj['box']:
                cur_frame = str(int(info['-frame']) + 1)
                bb_left = int(float(info['-xtl']))
                bb_top = int(float(info['-ytl']))
                bb_width = int(float(info['-xbr'])) - int(float(info['-xtl']))
                bb_height = int(float(info['-ybr'])) - int(float(info['-ytl']))
                file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames


def preprocessvideo_v_4(video_name, json_name, output_dir, cls):
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
            if obj_label not in cls:
                continue
            for info in obj['box']:
                cur_frame = str(int(info['_frame']) + 1)
                bb_left = int(float(info['_xtl']))
                bb_top = int(float(info['_ytl']))
                bb_width = int(float(info['_xbr'])) - int(float(info['_xtl']))
                bb_height = int(float(info['_ybr'])) - int(float(info['_ytl']))
                file.append([cur_frame, idx, bb_left, bb_top, bb_width, bb_height, 1, 1, 1])
            idx += 1
        with open(ops.join(outtxt_dir, 'gt.txt'), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for d in file:
                writer.writerow(d)
    return outimg_dir, outtxt_dir, frame_count, all_frames
