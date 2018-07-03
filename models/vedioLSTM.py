# coding:utf8
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from .BasicModule import BasicModule


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class vedioLSTM(BasicModule):
    def __init__(self, cfg, encoder=None):
        super(vedioLSTM, self).__init__()
        self.encoder = encoder
        self.nlayers = cfg.num_layers
        self.hid_size = cfg.hidden_size
        self.num_bis = 1
        self.maxk = cfg.kmax_pooling
        if cfg.bidirectional:
            self.num_bis = 2

        self.lstm = t.nn.LSTM(input_size=cfg.encode_fc,
                              hidden_size=cfg.hidden_size,
                              num_layers=cfg.num_layers,
                              bias=True,
                              batch_first=False,
                              dropout=cfg.dropout,
                              bidirectional=cfg.bidirectional
                              )
        self.fc = nn.Sequential(
            nn.Linear(self.num_bis * cfg.hidden_size * self.maxk, cfg.linear_hidden_size),  # 使用所有的H输出
            # nn.Linear(2 * cfg.hidden_size, cfg.linear_hidden_size),  # 使用H-last作为分类输出[bs,512]
            # nn.BatchNorm1d(cfg.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.linear_hidden_size, cfg.n_classes)
        )
        self.hidden = self.init_hidden(cfg.batch_size)

    def init_hidden(self, bsz):
        return [Variable(t.zeros(self.nlayers * self.num_bis, bsz, self.hid_size)).cuda(),
                Variable(t.zeros(self.nlayers * self.num_bis, bsz, self.hid_size)).cuda()]

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x, sv_out = self.encoder(x)
        self.hidden = self.init_hidden(x.shape[0])
        x = self.lstm(x.permute(1, 0, 2), self.hidden)[0].permute(1, 2, 0)  # 使用所有的H输出
        # x = self.lstm(x.permute(1, 0, 2), self.hidden)[0][-1]  # 使用H-last作为分类输出[bs,512]
        # print('before kmax:', x.shape) #before kmax:', (32, 256, 14)
        x = kmax_pooling(x, 2, self.maxk)  # 使用所有的H输出[bs, hid_size*2, 2]
        # print('after kmax:', x.shape)  #('after kmax:', (32, 256, 2))
        x = self.fc(x.view(x.size(0), -1))
        return x, sv_out
