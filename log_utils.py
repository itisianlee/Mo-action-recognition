# coding:utf8
import time
import os


def get_log_dir(logdir='logdir', name='vedioLSTM', flag='icpv4'):
    tim_flag = time.strftime("-%m-%d-%H-%M", time.localtime())
    flag = name + '-' + flag + tim_flag
    path = os.path.join(logdir, flag)
    # os.mkdir(path)
    return path
