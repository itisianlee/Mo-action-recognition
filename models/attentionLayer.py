# coding:utf8
import torch
import torch.nn as nn
from .BasicModule import BasicModule


class AttentionLayer(BasicModule):
    def __init__(self, cfg):
        super(AttentionLayer, self).__init__()
        self.a_softmax = nn.Softmax(dim=1)
        self.W_layer = nn.Linear(2 * cfg.hidden_size, 2 * cfg.hidden_size)
        self.U_layer = nn.Linear(2 * cfg.hidden_size, 1)
        self.hidden_size = cfg.hidden_size
        if cfg.bidirectional:
            self.num_bis = 2
        self.sequence = cfg.frames

    def forward(self, x_hid):
        x_hid_ = x_hid.view(-1, self.hidden_size * self.num_bis)
        u = self.W_layer(x_hid_)
        attn = self.U_layer(u)
        attn = attn.view(self.sequence, -1)
        alpha = self.a_softmax(attn)
        alpha = alpha.unsqueeze(2)
        attn_vec = torch.sum(x_hid * alpha, 0)
        return attn_vec
