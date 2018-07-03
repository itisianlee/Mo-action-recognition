# coding:utf8
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import nn
from torch.optim import lr_scheduler

from datasets import FaceRecognition, train_spatial_transform, \
    TemporalRandomCrop, val_spatial_transform
from tensorboard_logger import Logger
from log_utils import get_log_dir
from train import train_epoch
from validation import val_epoch
from models import get_end_net, get_encoder_net, CNNencoder
from config import cfg
from opts import parse_opts


def main():
    opt = parse_opts()

    ecd_name, cls_name = opt.model_name.split('-')
    ecd_model = get_encoder_net(ecd_name)
    cls_model = get_end_net(cls_name)

    cfg.encoder_model = ecd_name
    cfg.classification_model = cls_name

    if opt.debug:
        cfg.debug = opt.debug
    else:
        if opt.tensorboard == 'TEST':
            cfg.tensorboard = opt.model_name
        else:
            cfg.tensorboard = opt.tensorboard
            cfg.flag = opt.flag
    model = cls_model(cfg, encoder=CNNencoder(cfg, ecd_model(pretrained=True)))
    cfg.video_path = os.path.join(cfg.root_path, cfg.video_path)
    cfg.annotation_path = os.path.join(cfg.root_path, cfg.annotation_path)

    cfg.list_all_member()

    torch.manual_seed(cfg.manual_seed)
    print('##########################################')
    print('####### model 仅支持单GPU')
    print('##########################################')
    model = model.cuda()
    print(model)
    criterion = nn.CrossEntropyLoss()
    if cfg.cuda:
        criterion = criterion.cuda()

    print('##########################################')
    print('####### train')
    print('##########################################')

    training_data = FaceRecognition(cfg,
                                    '/share5/public/lijianwei/faces/',
                                    TemporalRandomCrop(14),
                                    train_spatial_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.n_threads,
        drop_last=False,
        pin_memory=True)
    optimizer = model.get_optimizer(lr1=cfg.lr, lr2=cfg.lr2)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=cfg.lr_patience)
    print('##########################################')
    print('####### val')
    print('##########################################')
    validation_data = FaceRecognition(cfg,
                                      '/share5/public/lijianwei/faces/',
                                      TemporalRandomCrop(14),
                                      val_spatial_transform,
                                      phase='val')
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.n_threads,
        drop_last=False,
        pin_memory=True)
    print('##########################################')
    print('####### run')
    print('##########################################')
    path = None
    best_avg = 0.
    if cfg.debug:
        logger = None
    else:
        path = get_log_dir(cfg.logdir, name=cfg.tensorboard, flag=cfg.flag)
        logger = Logger(logdir=path)
        cfg.save_config(path)

    for i in range(cfg.begin_epoch, cfg.n_epochs + 1):
        train_epoch(i, train_loader, model, criterion, optimizer, cfg, logger)
        validation_loss = val_epoch(i, val_loader, model, criterion, cfg, logger, path, best_avg)
        scheduler.step(validation_loss)


if __name__ == '__main__':
    main()
