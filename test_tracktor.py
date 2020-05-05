import os
import time
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment

from src.tracktor.frcnn_fpn import FRCNN_FPN
from src.tracktor.config import get_output_dir
from src.tracktor.datasets.factory import COSMOSTestDataset
from src.tracktor.tracker import Tracker
from src.tracktor.reid.resnet import resnet50
from src.tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums

ex = Experiment()
ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])

@ex.automain
def main(tracktor, reid, _config, _log, _run):
    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    # output_dir = /Users/smiffy/Documents/GitHub/Tracktor_COSMOS/output/tracktor/COSMOS/Tracktor++
    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')  # only used in eval

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector.")
    obj_detect_p = FRCNN_FPN(num_classes=2)
    obj_detect_v = FRCNN_FPN(num_classes=2)

    obj_detect_p.load_state_dict(torch.load(_config['tracktor']['obj_detect_model_p'],
                                          map_location=lambda storage, loc: storage))
    obj_detect_v.load_state_dict(torch.load(_config['tracktor']['obj_detect_model_v'],
                                            map_location=lambda storage, loc: storage))

    obj_detect_p.eval()
    obj_detect_p.cuda()
    obj_detect_v.eval()
    obj_detect_v.cuda()

    # reid
    reid_network_p = resnet50(pretrained=False, **reid['cnn'])
    reid_network_p.load_state_dict(torch.load(tracktor['reid_weights'],
                                            map_location=lambda storage, loc: storage))
    reid_network_v = resnet50(pretrained=False, **reid['cnn'])
    reid_network_v.load_state_dict(torch.load(tracktor['reid_weights'],
                                            map_location=lambda storage, loc: storage))
    reid_network_p.eval()
    reid_network_p.cuda()
    reid_network_v.eval()
    reid_network_v.cuda()

    tracker_p = Tracker(obj_detect_p, reid_network_p, tracktor['tracker'])
    tracker_v = Tracker(obj_detect_v, reid_network_v, tracktor['tracker'])

    time_total = 0
    num_frames = 0
    dataset = []
    dataset.append(
        COSMOSTestDataset(tracktor['dataset_pathes']['root_track_7'], tracktor['dataset_pathes']['det_root_txt_7']))
    dataset.append(
        COSMOSTestDataset(tracktor['dataset_pathes']['root_track_8'], tracktor['dataset_pathes']['det_root_txt_8']))
    # dataset.append(
    #     COSMOSTestDataset(tracktor['dataset_pathes']['root_track_3'], tracktor['dataset_pathes']['det_root_txt_3']))
    dataset.append(
        COSMOSTestDataset(tracktor['dataset_pathes']['root_track_9'], tracktor['dataset_pathes']['det_root_txt_9']))
    dataset.append(
        COSMOSTestDataset(tracktor['dataset_pathes']['root_track_10'], tracktor['dataset_pathes']['det_root_txt_10']))
    # dataset.append(
    #    COSMOSTestDataset(tracktor['dataset_pathes']['root_track_6'], tracktor['dataset_pathes']['det_root_txt_6']))

    for seq in dataset:
        tracker_p.reset()
        tracker_v.reset()
        start = time.time()
        _log.info(f"Tracking: {seq}")
        data_loader = DataLoader(seq, batch_size=1, shuffle=False)

        for i, frame in enumerate(tqdm(data_loader)):
            if len(seq) * tracktor['frame_split'][0] <= i <= len(seq) * tracktor['frame_split'][1]:
                tracker_p.step(frame, cls='pedestrian')
                tracker_v.step(frame, cls='vehicle')
                num_frames += 1
        results_p = tracker_p.get_results()
        results_v = tracker_v.get_results()
        time_total += time.time() - start

        _log.info(f"Runtime for {seq}: {time.time() - start :.1f} s.")
        _log.info(f"Writing predictions to: {output_dir}")
        seq.write_results(results_p, results_v, output_dir)

        _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
                  f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")
