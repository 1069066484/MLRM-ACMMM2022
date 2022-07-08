import random
import numpy as np
from torch.utils.data import Dataset as torchDataset
np.random.seed(0)
import torch.nn.functional as F
from torchvision import transforms
import cv2
from package.dataset.data_cmt import *
try:
    import gensim
except:
    pass



SK = 0
IM = 1


class ToUint8(object):
    # randomly move the sketch
    def __init__(self):
        pass

    def __call__(self, im):
        try:
            return im.astype(np.uint8) * 255
        except:
            return im

    def __repr__(self):
        return self.__class__.__name__ + 'ToUint8'


class RandomShift(object):
    # randomly move the sketch
    def __init__(self, maxpix=32):
        self.maxpix = maxpix
        self.fill = (255,255,255)
        self.padding_mode = 'constant'

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        l = int(random.random() * self.maxpix)
        t = int(random.random() * self.maxpix)
        size = img.size
        img = F.pad(img, (l, t, self.maxpix - l, self.maxpix - t), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(img, size)
        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(maxpix={0})'.format(self.maxpix)


class PngExp(object):
    def __init__(self):
        pass

    def __call__(self, single_ch):
        if isinstance(single_ch, np.ndarray):
            return np.stack([single_ch, single_ch, single_ch], -1)
        return torch.cat([single_ch, single_ch, single_ch])

    def __repr__(self):
        return self.__class__.__name__ + 'PngExp'


def npfn(fn):
    if not fn.endswith('.npy'):
        fn += '.npy'
    return fn


def expand3(sk):
    if len(sk.shape) == 3:
        return sk
    return np.stack([sk, sk, sk], -1)


from PIL import Image
from package.dataset import pix2vox_trans
from torchvision.transforms import functional


class SBIRMultiCrop(transforms.FiveCrop):
    def __init__(self, size):
        super().__init__(size=size)

    def forward(self, img):
        return functional.five_crop(img, self.size) + functional.five_crop(img.flip(2), self.size)


def make_globals(sz=256, target_size=56, crop_size=224, aug_params="{'degrees': 5, 'scale':0.1, 'shear': 5}"):
    global IM_SIZE, trans_norm_op, trans_noaug_l, trans_noaug, trans_noaug_single_ch, random_permuteRGB, pil_resize, \
        random_color_jitter, tensor_norm, trans_aug_simple, trans_aug_huge, trans_aug_simple_sk, \
        trans_aug_adv_sk, trans_aug_huge_sk, trans_aug_adv, trans_noaug_crop,\
        trans_noaug_sk_imp4, trans_aug_sk3_ret4, TARGET_SIZE

    if isinstance(aug_params, str):
        aug_params = eval(aug_params)
    if 'rgb' not in aug_params:
        aug_params['rgb'] = 0.6

    IM_SIZE = sz
    TARGET_SIZE = target_size

    trans_norm_op = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    trans_noaug_l = [transforms.ToPILImage(),
                                      transforms.Resize([crop_size, crop_size]),
                                      transforms.ToTensor(),
                                      trans_norm_op
                                      ]
    trans_noaug = transforms.Compose(trans_noaug_l)

    trans_noaug_crop_l = [transforms.ToPILImage(),
                                      transforms.Resize([int(crop_size * 1.1), int(crop_size * 1.1)]),
                                      transforms.ToTensor(),
                                      trans_norm_op,
                                        SBIRMultiCrop(crop_size)
                                      ]
    trans_noaug_crop = transforms.Compose(trans_noaug_crop_l)


    random_permuteRGB = transforms.RandomApply([pix2vox_trans.RandomPermuteRGB()], p=aug_params['rgb'])
    pil_resize = [transforms.ToPILImage(), transforms.Resize((IM_SIZE, IM_SIZE,))]
    random_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * 1, 0.4 * 1, 0.4 * 1, 0.2 * 1)], p=0.6)
    tensor_norm = [transforms.ToTensor(), trans_norm_op]

    crops = [transforms.RandomCrop(crop_size)]
    if crop_size + 2 >= IM_SIZE:
        crops = []

    trans_aug_simple = [
        random_permuteRGB,
        *pil_resize,
        *crops,
        random_color_jitter,
        transforms.RandomApply([transforms.RandomAffine(degrees=3, translate=(0.0, 0.0), scale=(0.95, 1.05), shear=(3, 3),
                                                        fillcolor=(255,255,255))], p=0.5),
        *tensor_norm
    ]
    trans_aug_adv = [
        random_permuteRGB,
        *pil_resize,
        *crops,
        random_color_jitter,
        transforms.RandomApply([transforms.RandomAffine(degrees=aug_params['degrees'],
                                                        translate=(0.05, 0.05),
                                                        scale=(1.0 - aug_params['scale'], 1.0 + aug_params['scale']),
                                                        shear=aug_params['shear'],
                                                        fillcolor=(255,255,255))],
                               p=0.8),
        *tensor_norm
    ]
    trans_aug_huge = [
        random_permuteRGB,
        pil_resize[0],
        transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.8,1.0))], p=0.6),
        pil_resize[1],
        random_color_jitter,
        transforms.RandomApply([transforms.RandomAffine(degrees=25, translate=(0.15, 0.15), scale=(0.7, 1.3), shear=(20, 20), fillcolor=(255,255,255))], p=0.6),
        *tensor_norm
    ]


    # trans_aug_single_ch = transforms.Compose([PngExp(), *trans_aug_l])

    # in, poses of objects almost remain
    trans_aug_simple_sk = transforms.Compose([t for t in trans_aug_simple if id(t) != id(random_color_jitter)])
    trans_aug_simple = transforms.Compose(trans_aug_simple)

    # a little pose bias
    trans_aug_adv_sk = transforms.Compose([t for t in trans_aug_adv if id(t) != id(random_color_jitter)])
    trans_aug_adv = transforms.Compose(trans_aug_adv)

    # simple indicating the same image, huge transformation
    trans_aug_huge_sk = transforms.Compose([t for t in trans_aug_huge if id(t) != id(random_color_jitter)])
    trans_aug_huge = transforms.Compose(trans_aug_huge)


make_globals()


def make_trans_rec(sz=None):
    if sz is None:
        sz = IM_SIZE
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((sz, sz,)),
        transforms.ToTensor(),
    ])

to_tensor = transforms.ToTensor()


def _load_all_ims(folder, single_ch=False, sz=IM_SIZE, continue_cond=lambda x: False, ret_f=False, lim=99999999,
                  post_fix=["jpg","jpeg","png"], edge=1, buf_folder=None, logger=None):
    _print = lambda x: print(x) if logger is None else logger.info(x)
    ims = []
    files = []
    fs = [f for f in os.listdir(folder) if f.split(".")[-1] in post_fix and not continue_cond(f)]
    _print("Going to process {} files".format(len(fs)))
    for idx, f in enumerate(sorted(fs)):

        if logger is not None and (idx + 1) % 500 == 0:
            logger.info("Loading {} files in {}".format(idx+1, folder))
        f0 = f
        files.append(f)
        f = join(folder, f)

        if edge is None:
            im = cv2.imread(f)
            ims.append(cv2.resize(im[:, :, 0] if single_ch else im[:, :, ::-1], (sz, sz)))
        else:
            try:
                arrCls = ArrCls(
                       f, d=join(buf_folder, f0.split(".")[-2]) if buf_folder is not None else None, sz=None, tmp=True
                    )
            except Exception as e:
                _print(str(e))
                files.remove(files[-1])
                continue
            ims.append(arrCls)
            # logger.info(str(arrCls.info()))
        if len(files) == lim: break
    if edge is None:
        ims = np.array(ims)
        _print("Get {}".format(ims.shape))
    else:
        _print("Get {}".format(len(ims)))
    if ret_f:
        return ims, [join(folder, f) for f in files]
    return ims


class FG_dataloader(torchDataset):
    def __init__(self, logger=None, sz=None, cls_aug=0):
        super(FG_dataloader, self).__init__()
        self.logger = logger
        sz = sz if sz is not None else IM_SIZE
        self.sz = sz
        self.cls_aug = cls_aug

    def _print(self, debug):
        try:
            logger = self.logger
            print(debug) if logger is None else logger.info(debug)
        except:
            print(debug)

    def print(self, debug):
        logger = self.logger
        print(debug) if logger is None else logger.info(debug)


    def _flip(self, t, tp=0):
        if tp is None: tp = np.random.randint(0,4)
        if tp == 0: return t
        if tp == 1: return t[:, ::-1, ...]
        if tp == 2: return t[::-1, ...]
        return t[::-1, ::-1]

    def _h_flip(self, t):
        return t[:, ::-1, ...]

    def _v_flip(self, t):
        return t[::-1, ...]
