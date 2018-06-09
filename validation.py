# coding:utf8
from torch.autograd import Variable
import torch as t

from utils import AverageMeter, calculate_accuracy
from tqdm import tqdm


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

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
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

        step = i + 1 + (epoch - 1) * len(data_loader)
        if step % opt.step_every_summary == 0:
            # logger.log_value('Val_Loss', loss.data[0], step)
            logger.log_value('Val_Loss', loss.item(), step)
            logger.log_value('Val_Accuracy', acc, step)
        print("val.size:", inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

    logger.log_value('Val_Loss_avg_epoch', losses.avg, epoch)
    logger.log_value('Val_Accuracy_avg_epoch', accuracies.avg, epoch)

    return losses.avg
