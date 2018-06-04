# coding:utf8
from .resnet import resnet18


class CNNencoder():
    def __init__(self, batch_size=2, frames=16, img_size=(224, 224), net=resnet18(True)):
        self.net = net
        self.frames = frames
        self.batch_size = batch_size
        self.size = img_size

    def __call__(self, input):
        return self.encoding(input)

    def encoding(self, input):
        x = input.view(-1, 3, self.size[0], self.size[1])
        x = self.net(x)
        return x.view(self.batch_size, self.frames, -1)
