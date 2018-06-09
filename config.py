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

    n_classes = 101  # 类别

    batch_size = 16
    frames = 16
    img_size = (112, 112)
    net = resnet18(pretrained=True)

    encode_dim = 512
    hidden_size = 256  # LSTM hidden size
    num_layers = 2  # LSTM layers
    linear_hidden_size = 1024  # 全连接层隐藏元数目

    lr = 1e-3  # 学习率
    lr2 = 1e-6  # encoder层的学习率

    # ==========================
    # opts.py中的配置
    # ==========================
    root_path = '/home/lijianwei/share5/action_datasets'
    video_path = 'ucf101'
    annotation_path = 'ucfTrainTestlist/ucf101_01.json'
    checkpoints = '/home/lijianwei/share5/Mo-action-recognition/checkpoints'
    logdir = '/home/lijianwei/share5/Mo-action-recognition/logdir'

    scales = [1.0, 0.84089641525, 0.7071067811803005, 0.5946035574934808, 0.4999999999911653]
    sample_size = 112
    sample_duration = 16
    norm_value = 1

    n_threads = 4

    manual_seed = 1
    cuda = True
    train_crop = 'corner'
    n_val_samples = 3

    lr_patience = 10
    begin_epoch = 1
    n_epochs = 200
    epoch_every_save_model = 5
    step_every_summary = 10

    def list_all_member(self):
        print('##########################################')
        print('####### config')
        print('##########################################')
        for k in dir(self):
            if not k.startswith('_') and k != 'list_all_member' and k != 'net':
                print(k, '-->', getattr(self, k))


cfg = Config()
