# coding:utf8
from __future__ import division
from __future__ import print_function

from models.CNNencoder import CNNencoder
from models.vedioLSTM import vedioLSTM
from config import cfg

import os
import torch
from torch import nn
from torch.optim import lr_scheduler
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from dataset import get_training_set, get_validation_set
from utils import Logger
from log_utils import get_log_dir
from train_log import train_epoch
from validation_log import val_epoch

if __name__ == '__main__':
    cfg.video_path = os.path.join(cfg.root_path, cfg.video_path)
    cfg.annotation_path = os.path.join(cfg.root_path, cfg.annotation_path)

    cfg.list_all_member()

    # with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
    #     json.dump(vars(opt), opt_file)

    torch.manual_seed(cfg.manual_seed)

    # model, parameters = generate_model(opt)
    model = vedioLSTM(cfg, encoder=CNNencoder(cfg))
    print('##########################################')
    print('####### model 仅支持单GPU')
    print('##########################################')
    model = model.cuda()
    print(model)
    criterion = nn.CrossEntropyLoss()
    if cfg.cuda:
        criterion = criterion.cuda()

    norm_method = Normalize([0, 0, 0], [1, 1, 1])

    print('##########################################')
    print('####### train')
    print('##########################################')
    assert cfg.train_crop in ['random', 'corner', 'center']
    if cfg.train_crop == 'random':
        crop_method = (cfg.scales, cfg.sample_size)
    elif cfg.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(cfg.scales, cfg.sample_size)
    elif cfg.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            cfg.scales, cfg.sample_size, crop_positions=['c'])
    spatial_transform = Compose([
        crop_method,
        RandomHorizontalFlip(),
        ToTensor(cfg.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(cfg.sample_duration)
    target_transform = ClassLabel()
    training_data = get_training_set(cfg, spatial_transform,
                                     temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.n_threads,
        drop_last=True,
        pin_memory=True)
    train_logger = Logger(
        os.path.join(cfg.custom_logdir, 'train.log'),
        ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(
        os.path.join(cfg.custom_logdir, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    optimizer = model.get_optimizer(lr1=cfg.lr, lr2=cfg.lr2)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=cfg.lr_patience)
    print('##########################################')
    print('####### val')
    print('##########################################')
    spatial_transform = Compose([
        Scale(cfg.sample_size),
        CenterCrop(cfg.sample_size),
        ToTensor(cfg.norm_value), norm_method
    ])
    temporal_transform = LoopPadding(cfg.sample_duration)
    target_transform = ClassLabel()
    validation_data = get_validation_set(
        cfg, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.n_threads,
        drop_last=False,
        pin_memory=True)
    val_logger = Logger(
        os.path.join(cfg.custom_logdir, 'val.log'), ['epoch', 'loss', 'acc'])

    print('##########################################')
    print('####### run')
    print('##########################################')
    path = get_log_dir(cfg.logdir, name=cfg.model)

    for i in range(cfg.begin_epoch, cfg.n_epochs + 1):
        train_epoch(i, train_loader, model, criterion, optimizer, cfg, train_logger, train_batch_logger)
        validation_loss = val_epoch(i, val_loader, model, criterion, cfg, val_logger)

        scheduler.step(validation_loss)
