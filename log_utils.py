# coding:utf8
import time
import os


def get_log_dir(logdir='logdir', name='vedioLSTM'):
    tim_flag = time.strftime("-%m-%d-%H-%M", time.localtime())
    flag = name + tim_flag
    path = os.path.join(logdir, flag)
    # os.mkdir(path)
    return path
