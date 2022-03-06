import argparse
from pathlib import Path
import time
import json
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
import random

import util.misc as utils
from engine import train_one_epoch, evaluate, test
from models.backbone import PositionEmbeddingSine
from models.transformer import Transformer
from models.rddetr import RDDETR
from models.matcher import HungarianMatcher
from models.criterion import SetCriterion
from datasets.radar import build_dataset


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector using radar signal', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=1, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--trans_dropout', default=0.1, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set_ce_coef', default=1, type=float)
    parser.add_argument('--set_L1_coef', default=2, type=float)
    parser.add_argument('--set_giou_coef', default=3, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)

    parser.add_argument('--dataset_dir', default='/home/ciot/Desktop/Radar_signal_processing/keypoint_data/')
    parser.add_argument('--output_dir', default='./outputs/loss_testing_scaled_backbone')
    parser.add_argument('--imaging_output_dir',
                        default='/home/ciot/Desktop/Radar_signal_processing/result_image/loss_testing_scaled_backbone_test/')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume_checkpoint', default='./outputs/loss_testing_scaled_backbone/checkpoint.pth')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--iou_threshold', default=0.7, type=float)
    parser.add_argument('--result_imaging', default=True)
    return parser


def main(args):
    print(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    positional_encoding = PositionEmbeddingSine(hidden_dim=args.hidden_dim, max_length=10000,
                                                normalize=True, scale=None)
    transformer = Transformer(d_model=args.hidden_dim, dropout=args.trans_dropout, nhead=args.n_heads,
                              dim_feedforward=args.dim_feedforward, num_encoder_layers=args.enc_layers,
                              num_decoder_layers=args.dec_layers, normalize_before=args.pre_norm,
                              return_intermediate_dec=True)
    model = RDDETR(position_encoding=positional_encoding, transformer=transformer,
                   num_classes=args.num_classes, num_queries=args.num_queries, aux_loss=args.aux_loss)
    matcher = HungarianMatcher(cost_class=args.set_ce_coef, cost_keypoint=args.set_L1_coef, cost_giou=args.set_giou_coef)
    weight_dict = {'loss_keypoints': args.set_L1_coef, 'loss_giou': args.set_giou_coef}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['boxes']
    criterion = SetCriterion(num_classes=args.num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, test=False)
    criterion.to(device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(dataset='train', base_dir=args.dataset_dir)
    dataset_val = build_dataset(dataset='val', base_dir=args.dataset_dir)
    dataset_test = build_dataset(dataset='test', base_dir=args.dataset_dir)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=1)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=1)
    data_loader_test = DataLoader(dataset_test, batch_size=1, sampler=sampler_test, drop_last=False,
                                  collate_fn=utils.collate_fn, num_workers=1)

    output_dir = Path(args.output_dir)
    if args.resume_checkpoint:
        if args.resume_checkpoint.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume_checkpoint, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model=model, criterion=criterion, data_loader=data_loader_train,
                                      optimizer=optimizer, device=device, epoch=epoch, max_norm=args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        valid_stats = evaluate(model=model, criterion=criterion, data_loader=data_loader_val, device=device)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'valid_{k}': v for k, v in valid_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    test(model=model, criterion=criterion, data_loader=data_loader_test, device=device,
         iou_threshold=args.iou_threshold, imaging=args.result_imaging, output_dir=args.imaging_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RDDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.imaging_output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
