# coding:utf8
import torch as t


class CNNencoder(t.nn.Module):
    def __init__(self, cfg):
        super(CNNencoder, self).__init__()
        self.net = cfg.net
        self.frames = cfg.frames
        self.batch_size = cfg.batch_size
        self.size = cfg.img_size

    def forward(self, input):
        x = input.contiguous().view(-1, 3, self.size[0], self.size[1])
        x = self.net(x)
        return x.view(-1, self.frames, 512)
