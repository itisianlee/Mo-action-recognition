# coding:utf8
import time
import warnings
from models.resnet import resnet18

tfmt = '%m%d_%H%M%S'


class Config(object):
    loss = 'multilabelloss'
    encoder_type = 'resnet'
    model = 'vedioLSTM'

    kmax_pooling = 2  # k

    env = time.strftime(tfmt)  # Visdom env
    plot_every = 10  # 每10个batch，更新visdom等

    max_epoch = 100
    min_lr = 1e-5  # 当学习率低于这个值，就退出训练
    lr_decay = 0.99  # 当一个epoch的损失开始上升lr = lr*lr_decay
    weight_decay = 0  # 2e-5 # 权重衰减
    weight = 1  # 正负样本的weight
    decay_every = 3000  # 每多少个batch 查看一下score,并随之修改学习率

    num_classes = 101  # 类别

    batch_size = 2
    frames = 16
    img_size = (112, 112)
    net = resnet18(True)

    encode_dim = 512
    hidden_size = 256  # LSTM hidden size
    num_layers = 2  # LSTM layers
    linear_hidden_size = 1024  # 全连接层隐藏元数目

    lr = 1e-3  # 学习率
    lr2 = 1e-5  # embedding层的学习率


cfg = Config()
