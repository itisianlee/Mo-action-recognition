# coding:utf8
import torch as t
from models.CNNencoder import CNNencoder
from models.vedioLSTM import vedioLSTM
from config import cfg
from dataset import get_training_set, get_validation_set
import os


def main():
    cfg.video_path = os.path.join(cfg.root_path, cfg.video_path)
    cfg.annotation_path = os.path.join(cfg.root_path, cfg.annotation_path)
    training_data = get_training_set(cfg, None, None, None)
    validation_data = get_validation_set(cfg, None, None, None)
    print(len(training_data))
    print(len(validation_data))

    # vedionet = vedioLSTM(cfg, encoder=CNNencoder(cfg))
    # vedionet = vedionet.cuda()
    # # vedionet = t.nn.DataParallel(vedionet)
    # # print(vedionet)
    # input = t.autograd.Variable(t.randn(8, 3, 16, 112, 112).cuda())
    # net = vedionet(input)
    # print(net)
    # print(net(input))


if __name__ == '__main__':
    main()
