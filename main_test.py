# coding:utf8
import torch as t
from models.CNNencoder import CNNencoder
from models.vedioLSTM import vedioLSTM
from config import cfg


def main():
    vedionet = vedioLSTM(cfg, encoder=CNNencoder(cfg))
    print(vedionet)
    input = t.autograd.Variable(t.randn(2, 3, 16, 112, 112))
    net = vedionet(input)
    # print(net)
    # print(net(input))


if __name__ == '__main__':
    main()
