# coding:utf8
import torch
from torch.autograd import Variable
import time
import os
import sys
from tqdm import tqdm

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    for i, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

        print("train.size:", inputs.size(0))
        if opt.cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs.cuda())
        # print("outputs.type:", outputs)
        # print("targets.type:", targets)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        step = i + 1 + (epoch - 1) * len(data_loader)
        if step % opt.step_every_summary == 0:
            # logger.log_value('Train_Loss', loss.data[0], step)
            logger.log_value('Train_Loss', loss.item(), step)  # pytorch 0.5将有的
            logger.log_value('Train_Accuracy', acc, step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % opt.epoch_every_save_model == 0:
        save_file_path = os.path.join(opt.checkpoints,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
