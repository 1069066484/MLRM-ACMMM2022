try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


import os
import numpy as np
import datetime
from queue import Queue
import torch
nn = torch.nn


exists = os.path.exists
join = os.path.join


def curr_time_str():
    return datetime.datetime.now().strftime('%y%m%d%H%M%S')


def mkdir(dir):
    try:
        if not exists(dir):
            os.makedirs(dir)
        return dir
    except:
        return dir


def aspath(path):
    assert exists(path), "path {} should exist!".format(path)
    return path


def traverse(folder, postfix='', rec=False, only_file=True):
    """
    Traverse all files in the given folder
    :param folder: The name of the folder to traverse.
    :param postfix: Required postfix
    :param rec: recursively or not
    :param only_file: Do not yield folder
    :return: paths of the required files
    """
    q = Queue()
    q.put(aspath(folder))
    while not q.empty():
        folder = q.get()
        for path in os.listdir(folder):
            path = join(folder, path)
            if os.path.isdir(path):
                q.put(path)
                if only_file:
                    continue
            if path.endswith(postfix):
                yield path
        if not rec:
            break


import torch
import cv2


def cv2Wait():
    if hasattr(cv2, "waitKeyEx"):
        cv2.waitKeyEx()
    else:
        cv2.waitKey()


class BatchNorm1d234(nn.Module):
    def __init__(self, dim):
        super(BatchNorm1d234, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        if len(x.shape) == 3:
            return self.bn(x.permute(0, 2, 1)).permute(0,2,1)
        if len(x.shape) == 4:
            shape = x.shape
            return self.bn(x.reshape([-1, self.dim])).reshape(shape)
        raise Exception("BatchNorm1d234 input shape: {}".format(x.shape))


class MLP(nn.Module):
    def __init__(self, layer_szs, act=nn.ReLU(inplace=True), last_act=None, pre_act=None, bias=True, bn=True,
                 last_inplace=None, noise=None):
        super(MLP, self).__init__()
        self.layer_szs = layer_szs
        self.act = act
        self.last_act = last_act
        self.noise = noise
        try:
            if last_inplace is not None:
                self.last_act.inplace = last_inplace
        except:
            pass
        self.pre_act = pre_act
        self.bias = bias
        self.bn = bn
        self.linears = []
        self._make_layers()

    def _make_layers(self):
        modules = []
        if self.pre_act:
            modules.append(self.pre_act)
        lm1 = len(self.layer_szs) - 1
        for i in range(lm1):
            modules.append(nn.Linear(self.layer_szs[i], self.layer_szs[i+1], bias=self.bias))
            self.linears.append(modules[-1])
            if self.bn and i != lm1 - 1:
                modules.append(BatchNorm1d234(self.layer_szs[i+1]))
            if self.noise is not None and i != lm1 - 1:
                modules.append(GaussianNoiseLayer(std=self.noise))
            if self.act is not None and i != lm1 - 1:
                modules.append(self.act)

        if self.last_act is not None:
            if self.bn:
                modules.append(BatchNorm1d234(self.layer_szs[i+1]))
            modules.append(self.last_act)
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        return self.features(x)


class FakeFn(nn.Module):
    def __init__(self, fn=lambda x: x):
        super(FakeFn, self).__init__()
        self.fn = fn

    def forward(self, *x):
        return self.fn(*x)


def add0(x):
    return x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)


from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def init_cp(cp):
    cp_ = checkpoint if cp else lambda f, x: f(x)
    cp_seq_ = checkpoint_sequential if cp else lambda f, s, x: f(x)
    return cp_, cp_seq_


class SharedAE(nn.Module):
    def __init__(self, layer_szs, act=nn.ReLU(inplace=True), last_act=nn.ReLU(),
                 mid_act=nn.ReLU(), bn=True):
        super(SharedAE, self).__init__()
        self.layer_szs = layer_szs
        self.act = act
        self.last_act = last_act

        self.mid_act = mid_act

        self.ls = []
        for layer1, layer2 in zip(layer_szs[1:], layer_szs[:-1]):
            self.ls.append(nn.Linear(layer1, layer2, bias=False))
        self.enc_bns = []
        for layer in layer_szs[1:]:
            self.enc_bns.append(nn.BatchNorm1d(layer) if bn else FakeFn())
        self.dec_bns = []
        for layer in layer_szs[:-1][::-1]:
            self.dec_bns.append(nn.BatchNorm1d(layer) if bn else FakeFn())
        self.ls = nn.Sequential(*self.ls)
        self.enc_bns = nn.Sequential(*self.enc_bns)
        self.dec_bns = nn.Sequential(*self.dec_bns)

    def encode(self, x):
        for i, (l, bn) in enumerate(zip(self.ls, self.enc_bns)):
            x = bn(x @ l.weight)
            x = self.act(x) if i != len(self.ls) -1 else self.mid_act(x)
        return x

    def decode(self, x):
        for i, (l, bn) in enumerate(zip(self.ls[::-1], self.dec_bns)):
            x = bn(x @ l.weight.T)
            x = self.act(x) if i != len(self.ls) -1 else self.last_act(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inputs):
        return inputs.reshape(inputs.shape[0], -1)


class Triplet(nn.Module):
    def __init__(self, margin=10):
        super(Triplet, self).__init__()
        self.margin = margin
        self.cosine = nn.CosineSimilarity()
        self.p_d = nn.PairwiseDistance()

    def forward(self, p1, p2, n):
        p_d = self.p_d(p1, p2)
        # d_sk_imn = torch.clamp(self.p_d(sketch, image_n), max=self.margin3)
        n_d = torch.clamp(self.margin - self.p_d(p1, n), min=0)
        return (torch.mean(p_d) + torch.mean(n_d)) / 2


class Reshape(nn.Module):
    def __init__(self, shape=None):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        if self.shape is None:
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), *self.shape)
        return x


class FModule(nn.Module):
    def __init__(self, f=None):
        super(FModule, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


def num_params(model):
    return sum(x.numel() for x in model.parameters())


def chceck_params_rec(module, depth=3):
    D = depth + 1
    NUM = num_params(module)
    def chceck_params_(module, depth):
        if depth == 0: return None
        num = num_params(module)
        if num == 0: return None
        print('----' * (D - depth), " t:", type(module), " n:", num, " r:", round(num / NUM, 5))
        for child in module.children():
            chceck_params_(child, depth - 1)
    chceck_params_(module, depth)


def get_hotmap(raw_hotmap, shape, thresh=None):
    '''
    :param raw_hotmap: [w, h]
    :param shape:
    :param thresh: None or a float thresh
    :return: normalized expanded resized hotmap
    '''
    min_pix = np.min(raw_hotmap)
    max_pix = np.max(raw_hotmap)
    hotmap = (raw_hotmap - min_pix) / max(max_pix - min_pix, 1e-4)
    if thresh is not None:
        hotmap[hotmap >= thresh] = 1.0
        hotmap[hotmap < thresh] = 0.0
    hotmap = cv2.resize(np.stack([hotmap] * 3, -1), shape) # expand
    hotmap = cv2.applyColorMap(255-np.uint8(hotmap[:,:,0]*255), cv2.COLORMAP_JET)#/255
    hotmap = cv2.cvtColor(hotmap, cv2.COLOR_BGR2RGB) / 255
    return hotmap


def visual_hotmap(hotmap, im, w=0.75, im_scale=0.45):
    hotmap = get_hotmap(hotmap, (im.shape[1], im.shape[0]))
    # print(im.dtype);exit()
    # print(hotmap.shape, type(hotmap), np.min(hotmap), np.max(hotmap), im.shape, type(im), np.min(im), np.max(im))
    hotmap = hotmap * w + (1.0 - w)
    im = im * im_scale + 128
    im_attended = (im * hotmap).astype(im.dtype)
    return im_attended


class GaussianNoiseLayer(nn.Module):
    def __init__(self, mean=0.0, std=0.2):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std)
            if x.is_cuda:
                noise = noise.cuda()
            x = x + noise
        return x


class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()

    def forward(self, x):
        return F.normalize(x)


def batch_cnn(f,  xs,  batch_size):
    xss = []
    if not isinstance(xs, list):
        xs = [xs]
    for start in range(0,  len(xs[0]),  batch_size):
        if isinstance(xs, list):
            xs_batch = [x[start:batch_size + start] for x in xs]
        vgg_xs = f(*xs_batch)
        xss.append(vgg_xs)
    ret = torch.cat(xss)
    return ret


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))
    def forward(self,x):
        return x + self.dummy - self.dummy


def kronecker_product(mat1, mat2):
    out_mat = torch.ger(mat1.view(-1), mat2.view(-1))
    # 这里的(mat1.size() + mat2.size())表示的是将两个list拼接起来
    out_mat = out_mat.reshape(*(mat1.size() + mat2.size())).permute([0, 2, 1, 3])
    out_mat = out_mat.reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))
    return out_mat



import math
def angle(v1, v2=None, PI=False, full=False):
    if len(v1) == 2:
        v1 = [*v1[0], *v1[1]]
    if len(v2) == 2:
        v2 = [*v2[0], *v2[1]]
    if v2 is None:
        v2 = [0,0,0,0]
    assert len(v1) >= 4 and len(v2) >= 4, ""
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180 and not full:
            included_angle = 360 - included_angle
    if PI:
        included_angle = included_angle / 180 * math.pi
    return included_angle


def cv_show1(arr, sz=None, name="cv_show1", w=True, only_ret=False):
    """
    :param arr:  an array. torch.Tensor or numpy array. Ensure arr represents 3-dimension array.
    :param sz: size of the image.
    :return: None.
    """
    if isinstance(sz, int):
        sz = (sz, sz)
    if len(arr.shape) == 4:
        arr = arr[0]
    if isinstance(arr, torch.Tensor):
        arr_min = arr.min(-1)[0].min(-1)[0][:, None, None]
        arr_max = arr.max(-1)[0].max(-1)[0][:, None, None]
        arr = (arr - arr_min) / (arr_max - arr_min)
        arr = (arr * 255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    if sz is not None:
        arr = cv2.resize(arr, sz)
    elif arr.shape[0] > 800 and arr.shape[1] > 800:
        arr = cv2.resize(arr, (arr.shape[1] // 2, arr.shape[0] // 2))
    if only_ret: return arr
    cv2.imshow(name, arr)
    if w: cv2Wait()
    return arr


def hotmap_integration(im, hotmap, w=0.5, only_hotmap=False):
    """
    :param im: input image, numpy array, h*w*3, 0-255
    :param hotmap: input hotmap, numpy array, h*w
    :param w: weight of hotmap
    :return: image-integrated hotmap
    """
    hotmap = hotmap.astype(np.float)
    ma = hotmap.max()
    mi = hotmap.min()
    hotmap = (hotmap - mi) / (ma - mi)
    hotmap = cv2.resize(np.stack([hotmap] * 3, -1), (im.shape[1], im.shape[0]))
    # print(hotmap.max(), hotmap.min())
    hotmap = cv2.applyColorMap(255-np.uint8(hotmap[:,:,0]*255), cv2.COLORMAP_JET)#/255
    hotmap = cv2.cvtColor(hotmap, cv2.COLOR_BGR2RGB)/255
    if only_hotmap:
        return (hotmap * 255).astype(np.uint8)
    hotmap = hotmap * w + (1.0 - w)
    return (im * hotmap).astype(np.uint8)


try:
    import cairosvg
except:
    print("cairosvg import error")

def svg2png_all(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            name = os.path.join(root, name)
            if name.endswith(".svg"):
                cairosvg.svg2png(url=name, write_to=name.replace(".svg", ".png"))


def cv_edge_extract(im, bin_thresh=150, canny_thresh=150):
    # bin_thresh: 160-230
    if isinstance(im, str):
        im = cv2.imread(im)
    if isinstance(bin_thresh, int):
        bin_thresh = [bin_thresh, 255]
    if isinstance(canny_thresh, int):
        canny_thresh = [canny_thresh, 255]
    _, image1 = cv2.threshold(im, bin_thresh[0], bin_thresh[1], cv2.THRESH_BINARY)
    image2 = 255 - image1.copy()
    image2_3 = cv2.Canny(image2, canny_thresh[0], canny_thresh[1])
    return 255 - image2_3

