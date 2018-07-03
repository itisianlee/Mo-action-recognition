# coding:utf8
import torch
import torch.utils.data as data
from PIL import Image
import os
import functools
from tqdm import tqdm

try:
    import accimage
except ImportError:
    accimage = None


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def make_dataset(path, debug=True):
    with open(path, 'r') as f:
        data = f.readlines()
    data_len = len(data)
    if debug:
        data_len = 500
    dataset = []
    for i in tqdm(range(data_len)):
        video_path, label = data[i].strip().split('\t')
        if not os.path.exists(video_path):
            continue
        n_frames = len(os.listdir(video_path))
        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'label': int(label),
            'frame_indices': list(range(1, n_frames + 1))
        }
        dataset.append(sample)
    return dataset


class FaceRecognition(data.Dataset):
    def __init__(self,
                 cfg,
                 root_path,
                 t_transform=None,
                 s_transform=None,
                 get_loader=get_default_video_loader,
                 phase='train'):
        self.data = make_dataset(os.path.join(root_path, phase), debug=cfg.debug)
        self.t_transform = t_transform
        self.s_transform = s_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        if self.t_transform is not None:
            frame_indices = self.t_transform(frame_indices)

        clip = self.loader(path, frame_indices)
        if self.s_transform is not None:
            self.s_transform.randomize_parameters()
            clip = [self.s_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]['label']
        return clip, target

    def __len__(self):
        return len(self.data)
