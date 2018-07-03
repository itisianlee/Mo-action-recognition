# coding:utf8
from torch.autograd import Variable
from tqdm import tqdm

from utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, logger):
    # print('train at epoch {}'.format(epoch))

    model.train()

    for i, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):

        # print("train.size:", inputs.size(0))
        if opt.cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        sv_targets = targets.unsqueeze(0).repeat(opt.frames, 1).permute(1, 0).contiguous().view(-1)
        targets = Variable(targets)
        sv_targets = Variable(sv_targets)
        outputs, sv_out = model(inputs.cuda())
        loss = criterion(outputs, targets)
        sv_loss = criterion(sv_out, sv_targets)
        acc = calculate_accuracy(outputs, targets)
        step = i + 1 + (epoch - 1) * len(data_loader)
        if step % opt.step_every_summary == 0:
            if logger is not None:
                logger.log_value('Train_Loss', loss.item(), step)  # pytorch 0.5将有的
                logger.log_value('SV_train_Loss', sv_loss.item(), step)  # pytorch 0.5将有的
                logger.log_value('Train_Accuracy', acc, step)
        loss += sv_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if epoch % opt.epoch_every_save_model == 0:
    #     save_file_path = os.path.join(opt.checkpoints,
    #                                   'save_{}.pth'.format(epoch))
    #     states = {
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }
    #     torch.save(states, save_file_path)
