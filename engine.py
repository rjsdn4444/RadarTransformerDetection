import math
import os
import sys
from typing import Iterable
import numpy as np
# import tqdm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import torch

import util.misc as utils
from util.imaging import Imaging
from util.box_ops import box_cxcywh_to_xyxy, box_iou


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'

    for radar_signals, targets in metric_logger.log_every(data_loader, 10, header):
        radar_signals = radar_signals.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(radar_signals)
        loss_dict = criterion(predictions, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return valid_stats


@torch.no_grad()
def test(model, criterion, data_loader, device, iou_threshold, imaging, output_dir):
    model.eval()
    criterion.eval()
    criterion.test = True
    imaging_interval = 1

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for radar_signals, targets in metric_logger.log_every(data_loader, 10, header):
        radar_signals = radar_signals.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(radar_signals)
        loss_dict = criterion(predictions, targets)
        weight_dict = criterion.weight_dict

        src_keypoints = criterion.src_keypoints.squeeze()
        target_keypoints = criterion.target_keypoints.squeeze()
        src_keypoints = np.array(src_keypoints.cpu().detach())
        target_keypoints = np.array(target_keypoints.cpu().detach())
        src_keypoints = src_keypoints.reshape((-1, 3))
        target_keypoints = target_keypoints.reshape((-1, 3))

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if imaging_interval % 50 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for i in range(len(target_keypoints)):
                if i == 0:
                    ax.scatter3D(target_keypoints[i][0], target_keypoints[i][1], target_keypoints[i][2],
                                 c='Red', label='target', s=10)
                    ax.scatter3D(src_keypoints[i][0], src_keypoints[i][1], src_keypoints[i][2],
                                 c='Blue', label='prediction', s=10)
                else:
                    ax.scatter3D(target_keypoints[i][0], target_keypoints[i][1], target_keypoints[i][2], c='Red', s=10)
                    ax.scatter3D(src_keypoints[i][0], src_keypoints[i][1], src_keypoints[i][2], c='Blue', s=10)

            for i in range(0, 4):
                ax.plot([target_keypoints[i][0], target_keypoints[i + 1][0]],
                        [target_keypoints[i][1], target_keypoints[i + 1][1]],
                        [target_keypoints[i][2], target_keypoints[i + 1][2]], c='Red')
                ax.plot([src_keypoints[i][0], src_keypoints[i + 1][0]],
                        [src_keypoints[i][1], src_keypoints[i + 1][1]],
                        [src_keypoints[i][2], src_keypoints[i + 1][2]], c='Blue')
            ax.plot([target_keypoints[3][0], target_keypoints[5][0]], [target_keypoints[3][1], target_keypoints[5][1]],
                    [target_keypoints[3][2], target_keypoints[5][2]], c='Red')
            ax.plot([src_keypoints[3][0], src_keypoints[5][0]], [src_keypoints[3][1], src_keypoints[5][1]],
                    [src_keypoints[3][2], src_keypoints[5][2]], c='Blue')
            ax.plot([target_keypoints[3][0], target_keypoints[9][0]], [target_keypoints[3][1], target_keypoints[9][1]],
                    [target_keypoints[3][2], target_keypoints[9][2]], c='Red')
            ax.plot([src_keypoints[3][0], src_keypoints[9][0]], [src_keypoints[3][1], src_keypoints[9][1]],
                    [src_keypoints[3][2], src_keypoints[9][2]], c='Blue')
            for i in range(5, 8):
                ax.plot([target_keypoints[i][0], target_keypoints[i + 1][0]],
                        [target_keypoints[i][1], target_keypoints[i + 1][1]],
                        [target_keypoints[i][2], target_keypoints[i + 1][2]], c='Red')
                ax.plot([src_keypoints[i][0], src_keypoints[i + 1][0]],
                        [src_keypoints[i][1], src_keypoints[i + 1][1]],
                        [src_keypoints[i][2], src_keypoints[i + 1][2]], c='Blue')
            for i in range(9, 12):
                ax.plot([target_keypoints[i][0], target_keypoints[i + 1][0]],
                        [target_keypoints[i][1], target_keypoints[i + 1][1]],
                        [target_keypoints[i][2], target_keypoints[i + 1][2]], c='Red')
                ax.plot([src_keypoints[i][0], src_keypoints[i + 1][0]],
                        [src_keypoints[i][1], src_keypoints[i + 1][1]],
                        [src_keypoints[i][2], src_keypoints[i + 1][2]], c='Blue')
            # ax.set_xlim([0, 1])
            # ax.set_ylim([0, 1])
            # ax.set_zlim([0, 1])
            plt.legend()
            fig.savefig(output_dir + '{}.png'.format(imaging_interval))
        imaging_interval += 1
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
