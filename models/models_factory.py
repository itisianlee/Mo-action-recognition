from .videoGRU import videoGRU
from .vedioLSTM import vedioLSTM
from .resnet import resnet50, resnet18, resnet34
from .inceptionv4 import inceptionv4
from .mobilev2 import mobilenetv2

encoder_nets = {
    'resnet50': resnet50,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'inceptionv4': inceptionv4,
    'mobilenetv2': mobilenetv2

}

classification_nets = {
    'gru': videoGRU,
    'lstm': vedioLSTM
}


def get_encoder_net(name=None):
    if name is None:
        raise ValueError('Name of encoder network is None')
    if name not in encoder_nets:
        raise ValueError('Name of encoder network unknown %s' % name)
    return encoder_nets[name]


def get_end_net(name=None):
    if name is None:
        raise ValueError('Name of classification network is None')
    if name not in classification_nets:
        raise ValueError('Name of classification network unknown %s' % name)

    return classification_nets[name]
