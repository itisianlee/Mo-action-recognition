# coding:utf8
import os


class Config(object):
    ##########################################
    # Model info
    ##########################################
    loss = 'multilabelloss'
    encoder_model = 'resnet50'
    classification_model = 'lstm'
    tensorboard = 'r50-2lLSTM'  # resnet50-2layers-LSTM
    flag = 'no'
    pretrained_path = None
    debug = False
    ##########################################
    # Basic config
    ##########################################
    n_classes = 2  # 类别
    batch_size = 32
    # net = inceptionv4()
    lr = 1e-3  # 学习率
    lr2 = 1e-6  # encoder层的学习率
    weight_decay = 0  # 2e-5 # 权重衰减
    n_threads = 4
    cuda = True
    lr_patience = 10
    begin_epoch = 1
    n_epochs = 200
    epoch_every_save_model = 5
    step_every_summary = 10

    ##########################################
    # LSTM/GRU config
    ##########################################
    encode_dim = 2048  # inception 1536，resnet50 2048，resnet18|34 512
    encode_fc = 512
    hidden_size = 128  # LSTM hidden size
    num_layers = 1  # LSTM layers
    bidirectional = True
    dropout = 0

    ##########################################
    # LSTM-->FC layer config
    ##########################################
    kmax_pooling = 2  # k
    linear_hidden_size = 128  # 全连接层隐藏元数目

    ##########################################
    # Path config
    ##########################################
    root_path = '/home/lijianwei/share5/action_datasets'
    video_path = 'ucf101'
    annotation_path = 'ucfTrainTestlist/ucf101_01.json'
    checkpoints = '/home/lijianwei/share5/Mo-action-recognition/checkpoints'
    logdir = '/home/lijianwei/share5/Mo-action-recognition/logdir'
    custom_logdir = '/home/lijianwei/share5/Mo-action-recognition/custom_logdir'

    ##########################################
    # Dataset config
    ##########################################
    frames = 14
    img_size = (112, 112)
    # scales = [1.0, 0.84089641525, 0.7071067811803005, 0.5946035574934808, 0.4999999999911653]
    scales = [1.0, 0.84089641525, 0.7071067811803005]
    sample_size = 112
    sample_duration = 14
    norm_value = 1
    manual_seed = 1
    train_crop = 'corner'
    # train_crop = 'center'
    n_val_samples = 3

    def list_all_member(self):
        print('##########################################')
        print('####### config')
        print('##########################################')
        for k in dir(self):
            if not k.startswith('_') and k != 'list_all_member' and k != 'save_config':
                print(k, '-->', getattr(self, k))

    def save_config(self, path):
        info = []
        for k in dir(self):
            if not k.startswith('_') and k != 'list_all_member' and k != 'save_config':
                info.append("%s:%s\n" % (k, getattr(self, k)))
        with open(os.path.join(path, 'config.txt'), 'w') as f:
            f.writelines(info)


cfg = Config()
