# coding:utf8
from torch.autograd import Variable
import torch as t
import os
from tqdm import tqdm

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger, path=None, best_avg=None):
    # print('validation at epoch {}'.format(epoch))

    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

        if opt.cuda:
            targets = targets.cuda(async=True)
        # inputs = Variable(inputs.cuda(), volatile=True)
        with t.no_grad():
            inputs = Variable(inputs.cuda())
            # targets = Variable(targets, volatile=True)
            sv_targets = targets.unsqueeze(0).repeat(opt.frames, 1).permute(1, 0).contiguous().view(-1)
            targets = Variable(targets)
            sv_targets = Variable(sv_targets)
            outputs, sv_out = model(inputs)
            loss = criterion(outputs, targets)
            sv_loss = criterion(sv_out, sv_targets)
            acc = calculate_accuracy(outputs, targets)

        step = i + 1 + (epoch - 1) * len(data_loader)
        if step % opt.step_every_summary == 0:
            # logger.log_value('Val_Loss', loss.data[0], step)
            if logger is not None:
                logger.log_value('Val_Loss', loss.item(), step)
                logger.log_value('SV_Val_Loss', sv_loss.item(), step)
                logger.log_value('Val_Accuracy', acc, step)
        # print("val.size:", inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
    if logger is not None:
        logger.log_value('Val_Accuracy_avg_epoch', accuracies.avg, epoch)
    if best_avg is not None and accuracies.avg > best_avg and path:
        save_file_path = os.path.join(path, 'best.model')
        t.save(model.state_dict(), save_file_path)
    return losses.avg
