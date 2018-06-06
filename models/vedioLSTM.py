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
        self.lstm = t.nn.LSTM(input_size=cfg.encode_dim,
                              hidden_size=cfg.hidden_size,
                              num_layers=cfg.num_layers,
                              bias=True,
                              batch_first=False,
                              # dropout = 0.5,
                              bidirectional=True
                              )
        self.fc = nn.Sequential(
            nn.Linear(2 * (cfg.hidden_size * cfg.num_layers), cfg.linear_hidden_size),
            # nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.linear_hidden_size, cfg.n_classes)
        )
        self.hidden = self.init_hidden(cfg.batch_size)

    def init_hidden(self, bsz):
        return [Variable(t.zeros(self.nlayers * 2, bsz, self.hid_size)).cuda(),
                Variable(t.zeros(self.nlayers * 2, bsz, self.hid_size)).cuda()]

    def forward(self, input):
        x = input.permute(0, 2, 1, 3, 4)
        x = self.encoder(x)
        self.hidden = self.init_hidden(x.shape[0])
        x = self.lstm(x.permute(1, 0, 2), self.hidden)[0].permute(1, 2, 0)
        x = kmax_pooling(x, 2, 2)
        x = self.fc(x.view(x.size(0), -1))
        # print('-------------------fc begin---------------------')
        # print(x)
        # print('-------------------fc end---------------------')
        return x
