import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description='this is Train options')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug mode')
    parser.add_argument('--no-debug', dest='debug', action='store_false', help='Debug mode')
    parser.set_defaults(debug=True)
    parser.add_argument(
        '--tensorboard',
        default='TEST',
        type=str,
        help='Model name about tensorboard (resnet50-2layers-LSTM-> r50-2lLSTM| )')
    parser.add_argument(
        '--encoder_model',
        default='',
        type=str,
        help='encoder model path')
    parser.add_argument(
        '--flag',
        default='No',
        type=str,
        help='Flag infomation about tensorboard')

    parser.add_argument(
        '--model_name',
        default='resnet50-lstm',
        type=str,
        help='Model name, examples:(resnet50-lstm | resnet50-gru')
    args = parser.parse_args()

    return args
