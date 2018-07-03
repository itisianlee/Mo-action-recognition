# coding:utf8
import torch as t
import torch.nn as nn
from .BasicModule import BasicModule


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class videoGRU(BasicModule):
    def __init__(self, cfg, encoder=None):
        super(videoGRU, self).__init__()
        self.encoder = encoder
        self.nlayers = cfg.num_layers
        self.hid_size = cfg.hidden_size
        self.num_bis = 1
        if cfg.bidirectional:
            self.num_bis = 2

        self.gru = t.nn.GRU(input_size=cfg.encode_dim,
                            hidden_size=cfg.hidden_size,
                            num_layers=cfg.num_layers,
                            bias=True,
                            batch_first=False,
                            dropout=cfg.dropout,
                            bidirectional=cfg.bidirectional
                            )
        self.fc = nn.Sequential(
            nn.Linear(self.num_bis * (cfg.hidden_size * 2), cfg.linear_hidden_size),  # 使用所有的H输出
            # nn.Linear(2 * cfg.hidden_size, cfg.linear_hidden_size),  # 使用H-last作为分类输出[bs,512]
            # nn.BatchNorm1d(cfg.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.linear_hidden_size, cfg.n_classes)
        )
        self.hidden = self.init_hidden(cfg.batch_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers * self.num_bis, bsz, self.hid_size).cuda()
        # return [Variable(t.zeros(self.nlayers * self.num_bis, bsz, self.hid_size)).cuda(),
        #         Variable(t.zeros(self.nlayers * self.num_bis, bsz, self.hid_size)).cuda()]

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.encoder(x)
        self.hidden = self.init_hidden(x.shape[0])
        x = self.gru(x.permute(1, 0, 2), self.hidden)[0].permute(1, 2, 0)  # 使用所有的H输出
        # x = self.lstm(x.permute(1, 0, 2), self.hidden)[0][-1]  # 使用H-last作为分类输出[bs,512]
        x = kmax_pooling(x, 2, 2)  # 使用所有的H输出
        x = self.fc(x.view(x.size(0), -1))
        return x
