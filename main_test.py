# coding:utf8
import torch as t
# from models.CNNencoder import CNNencoder
# from models.vedioLSTM import vedioLSTM
# from config import cfg
# from dataset import get_training_set, get_validation_set
# import os
# from models.inceptionv4 import inceptionv4
from models.mobilev2 import MobileNetV2
from models.resnet3D import resnet34
from torch import nn


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('denseblock{}'.format(i))
        ft_module_names.append('transition{}'.format(i))
    ft_module_names.append('norm5')
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.module = resnet34(num_classes=400, sample_size=112,
                               sample_duration=14, shortcut_type='A')

    def forward(self, x):
        return self.module(x)


def main():
    # cfg.video_path = os.path.join(cfg.root_path, cfg.video_path)
    # cfg.annotation_path = os.path.join(cfg.root_path, cfg.annotation_path)
    # training_data = get_training_set(cfg, None, None, None)
    # validation_data = get_validation_set(cfg, None, None, None)
    # print(len(training_data))
    # print(len(validation_data))

    # vedionet = vedioLSTM(cfg, encoder=CNNencoder(cfg))
    # vedionet = vedionet.cuda()
    # # vedionet = t.nn.DataParallel(vedionet)
    # # print(vedionet)
    # input = t.autograd.Variable(t.randn(8, 3, 16, 112, 112).cuda())
    # print(vedionet)
    # net = vedionet(input)
    # print(net.shape)

    # 分类网络定义-------------------
    input = t.autograd.Variable(t.randn(1, 3, 14, 112, 112).cuda())
    # net = MobileNetV2(n_class=101, input_size=112).cuda()

    model = Model().cuda()
    pretrain = t.load('/home/lijianwei/share5/3D-ResNets-PyTorch/models/resnet-34-kinetics.pth')

    model.load_state_dict(pretrain['state_dict'])

    model.module.fc = nn.Linear(model.module.fc.in_features, 101)
    model.module.fc = model.module.fc.cuda()

    parameters = get_fine_tuning_parameters(model, 0)
    # model = MobileNetV2(n_class=101, input_size=112, width_mult=1.)
    # net.load_state_dict(t.load('./checkpoints/mobilenetv2_718.pth.tar'))
    # net = net.cuda()
    out = model(input)
    # print(model)
    print(out.shape)


if __name__ == '__main__':
    main()
