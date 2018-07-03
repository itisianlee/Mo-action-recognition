# coding:utf8
import torch as t


class CNNencoder(t.nn.Module):
    def __init__(self, cfg, net):
        super(CNNencoder, self).__init__()
        self.net = net
        self.frames = cfg.frames
        self.batch_size = cfg.batch_size
        self.size = cfg.img_size
        self.out_size = cfg.encode_fc
        self.fc = t.nn.Linear(cfg.encode_dim, self.out_size)
        self.sv = t.nn.Linear(self.out_size, cfg.n_classes)

    def forward(self, input):
        x = input.contiguous().view(-1, 3, self.size[0], self.size[1])
        x = self.net(x)
        x = self.fc(x)
        sv_out = self.sv(x)
        # print('resnet_output:', x.shape)
        return x.view(-1, self.frames, self.out_size), sv_out
