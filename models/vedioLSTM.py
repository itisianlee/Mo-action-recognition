# coding:utf8
import torch as t


class vedioLSTM(t.nn.Module):
    def __init__(self, encoder=None, encode_dim=256, hidden_size=256, num_layers=2):
        super(vedioLSTM, self).__init__()
        self.encoder = encoder
        self.lstm = t.nn.LSTM(input_size=encode_dim, \
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bias=True,
                              batch_first=False,
                              # dropout = 0.5,
                              bidirectional=True
                              )

    def forward(self, input):
        x = self.encoder(input)
        x = self.lstm(x)
        return x
