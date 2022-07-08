from package.dataset.data_afg_ssl import *
from package.dataset.data_cmt import *
from package.dataset import data_afg_ssl
import torch


make_globals()


to_tensor = transforms.ToTensor()


class ADRAM_dataloader_clean(FG_dataloader):
    def __init__(self, folder_top=None, logger=None, sz=None, train=True, aug=True,  skip=1, flip=0, vanilla=1):
        super(ADRAM_dataloader_clean, self).__init__(logger=logger, sz=sz)
        self.aug = aug
        self.skip = skip
        self.train = train
        self.flip = flip
        self.folder_top = folder_top
        self.n_flip = 4
        self.vanilla = vanilla
        self.return_data_idx = 0
        self.aux_mode = False

    def aug_sk(self, aug=None):
        if aug is None:
            aug = self.aug
        if aug == 1:
            return data_afg_ssl.trans_aug_adv_sk if np.random.rand() < 0.8 else data_afg_ssl.trans_noaug
        elif aug == 2:
            return data_afg_ssl.trans_aug_simple_sk
        elif aug == 3:
            return data_afg_ssl.trans_aug_huge_sk if self.return_data_idx % 2 == 0 else data_afg_ssl.trans_noaug
        elif aug == 0:
            return data_afg_ssl.trans_noaug
        else:
            raise Exception("aug_sk - Wrong aug param: aug={}".format(aug))

    def aug_im(self, aug=None):
        if aug is None:
            aug = self.aug
        if aug == 1:
            return data_afg_ssl.trans_aug_adv if np.random.rand() < 0.8 else data_afg_ssl.trans_noaug
        elif aug == 2:
            return data_afg_ssl.trans_aug_simple
        elif aug == 3:
            return data_afg_ssl.trans_aug_huge if self.return_data_idx % 2 == 0 else data_afg_ssl.trans_noaug
        elif aug == 0:
            return data_afg_ssl.trans_noaug
        else:
            raise Exception("aug_im - Wrong aug param: aug={}".format(aug))

    def epoch(self):
        pass

    def return_data(self, sk, im, flip=None, idx_sk=None, idx_im=None):
        if not self.aug:
            flip = 0

        if self.flip:
            if flip is None or self.flip == 2:
                flip = self._flip(None)

            # if self.aug == 3: self.return_data_idx = 1 - self.return_data_idx
            sk = self._flip(sk, flip)
            im = self._flip(im, flip)

        im = self.aug_im()(im)
        sk = self.aug_sk()(sk)

        return sk, im, idx_sk, idx_im

    def traverse_get_x_path_i(self, x, what, i):
        raise Exception("Not implemented!")

    def traverse(self, what, batch_size=16, skip=1):
        assert what == IM or what == SK, "CMT_dataloader.traverse: what must be IM({})/SK({}), but get {}" \
            .format(IM, SK, what)
        rets_ims = []; rets_ids = []
        paths = []
        for i, x in enumerate(self.skim[what]):
            if i % skip != 0: continue
            x, path, label = self.traverse_get_x_path_i(x, what, i)
            if self.vanilla:
                paths.append(path)  # trans_noaug_crop
                rets_ims.append(data_afg_ssl.trans_noaug(x))
                rets_ids.append(label)
            else:
                crops = list(data_afg_ssl.trans_noaug_crop(x))
                n_crop = len(crops)
                rets_ims += crops
                paths += [path] * n_crop
                rets_ids += [label] * n_crop
            if len(rets_ims) >= batch_size:
                yield torch.stack(rets_ims, dim=0), torch.tensor(rets_ids), paths
                rets_ims = []; rets_ids = []; paths = []
        if len(rets_ids) >= 1:
            yield torch.stack(rets_ims, dim=0), torch.tensor(rets_ids), paths


class ADRAM_dataloader_Sketchy(ADRAM_dataloader_clean):
    def __init__(self, folder_top=None,  logger=None, sz=None, train=True, aug=1,
                 skip=1, flip=0, vanilla=1):
        super(ADRAM_dataloader_Sketchy, self).__init__(folder_top=folder_top, logger=logger, sz=sz, train=train,
                                                       aug=aug, skip=skip, flip=flip, vanilla=vanilla)
        self.flip = flip
        self.skim = [[],[]]
        self.fg_label = [[], []]
        self.train = train
        folder_sk = join(folder_top, "sketch/tx_000000000000_ready")
        folder_im = join(folder_top, "photo/tx_000000000000_ready")
        clss = sorted(os.listdir(folder_sk))
        self.clss = clss
        self.cls_base_index = [[0],[0]]
        corresponds = [[],[]]
        for i, name in enumerate(clss):
            sks_folder = join(folder_sk, name)
            ims_folder = join(folder_im, name)
            data_of_name, correspond = self._get_data_from_ims(sks_folder=sks_folder,
                                                   ims_folder=ims_folder)
            correspond[SK] = [[ci + len(self.skim[IM]) for ci in c] for c in correspond[SK]]
            correspond[IM] = [[ci + len(self.skim[SK]) for ci in c] for c in correspond[IM]]
            corresponds[SK] += correspond[SK]
            corresponds[IM] += correspond[IM]
            self.skim[SK] += data_of_name[SK]
            self.skim[IM] += data_of_name[IM]
            self.cls_base_index[IM].append(len(self.skim[IM]))
            self.cls_base_index[SK].append(len(self.skim[SK]))
            self._print('class {}, im: {}, sk: {} disposed'.format(name, len(data_of_name[IM]), len(data_of_name[SK])))
        self.corresponds = corresponds
        self.fg_label[IM] = np.arange(0, len(self.skim[IM]))
        self.fg_label[SK] = np.array([corresponds[SK][i][0] for i in range(len(self.skim[SK]))])
        self._print('Dataset loaded from \n'
                    'folder_sk:{}\n'
                    'folder_im:{}\n'
                    'sk_len:{}, im_len:{}, cls:{}\n'.format(
            folder_sk, folder_im, len(self.skim[SK]), len(self.skim[IM]),
            len(self.clss)))
        self.true_idx = self.arrange_cls_idx(self.cls_base_index[IM])

    def epoch(self):
        if np.random.rand() < 0.3:
            self.true_idx = np.random.permutation(self.n_flip * len(self.fg_label[IM]))
        else:
            self.true_idx = self.arrange_cls_idx(self.cls_base_index[IM])

    def arrange_cls_idx(self, cls_base_index):
        total = sum(cls_base_index)
        cls_base_index_ext = []
        curr = 0
        for l in cls_base_index:
            for j in range(self.n_flip):
                cls_base_index_ext += [x + curr + j * total for x in np.random.permutation(l)]
            curr += l
        return cls_base_index_ext

    def num(self, what=IM):
        return len(self.fg_label[what])

    def _get_data_from_ims(self, sks_folder, ims_folder):
        # paths = set([path.split('-')[0] for path in sorted(os.listdir(sks_folder))])
        sk_files = sorted(os.listdir(sks_folder))
        im_files = sorted(os.listdir(ims_folder))

        data_of_name = [[join(sks_folder, path) for path in sk_files
                   if path[-3:].lower() in ['jpeg','jpg','png','jpg'] ],
                  [join(ims_folder, path) for path in im_files
                   if path[-3:].lower() in ['jpeg','jpg','png','jpg']  # and path.split('.')[0] in paths)
                   ]]

        ids_sk = [os.path.splitext(os.path.split(n)[-1])[0].split('-')[0] for n in data_of_name[0]]
        ids_im = [os.path.splitext(os.path.split(n)[-1])[0].split('-')[0] for n in data_of_name[1]]

        sids_im = sorted(list(set(ids_im)))

        sids_im = [i_im for i, i_im in enumerate(sids_im) if i % self.skip == 0]

        if self.train: sids_im = [i_im for i, i_im in enumerate(sids_im) if i % 10]
        else: sids_im = [i_im for i, i_im in enumerate(sids_im) if i % 10 == 0]

        data_of_name = [[data_of_name[0][i] for i in range(len(data_of_name[0])) if ids_sk[i] in sids_im],
                        [data_of_name[1][i] for i in range(len(data_of_name[1])) if ids_im[i] in sids_im]]
        # print(len(data_of_name[0]), len(data_of_name[1]))
        ids_sk = [os.path.splitext(os.path.split(n)[-1])[0].split('-')[0] for n in data_of_name[0]]
        ids_im = [os.path.splitext(os.path.split(n)[-1])[0].split('-')[0] for n in data_of_name[1]]

        correspond = [[[] for _ in range(len(ids_sk))], [[] for _ in range(len(ids_im))]]
        for i, n in enumerate(ids_sk):
            correspond[IM][ids_im.index(n)].append(i)
            correspond[SK][i].append(ids_im.index(n))
        correspond = correspond
        return data_of_name, correspond

    def __getitem__(self, idx):
        idx = self.true_idx[idx]
        idx_im = idx % len(self.fg_label[IM])
        flip = idx_im // len(self.fg_label[IM])
        # print(self.corresponds[IM][idx_im], IM, idx_im)
        idx_sk = np.random.choice(self.corresponds[IM][idx_im])
        im = cv2.imread(self.skim[IM][idx_im])
        sk = expand3(cv2.imread(self.skim[SK][idx_sk]))
        ret = self.return_data(sk, im, flip, idx_sk, idx_im)
        return ret

    def traverse_get_x_path_i(self, x, what, i):
        x = cv2.imread(x)
        x = expand3(x)
        path = self.skim[what][i]
        return x, path, self.fg_label[what][i]

    def __len__(self):
        return len(self.fg_label[IM]) * self.n_flip if self.flip == 1 else len(self.fg_label[IM])


from package.dataset.data_afg_ssl import _load_all_ims


class ADRAM_dataloader_multi(ADRAM_dataloader_clean):
    def __init__(self, folder_top=None, logger=None, sz=None, train=True, aug=True, flip=0, vanilla=1):
        super(ADRAM_dataloader_multi, self).__init__(folder_top=folder_top, logger=logger, sz=sz, train=train, aug=aug,
                                                     flip=flip, vanilla=vanilla)
        self.skim = [[],[]]
        self._build_skim()
        self.print("sk: {},\t\tim: {}".format(self.skim[SK].shape, self.skim[IM].shape))

    def _build_skim(self):
        lines = []
        split = 'train' if self.train else 'test'
        with open(join(self.folder_top, 'photo_{}.txt'.format(split))) as f: lines += f.readlines()
        with open(join(self.folder_top, 'sketch_{}.txt'.format(split))) as f: lines += f.readlines()
        lines = [l.strip().split('.')[0] for l in lines if l.strip() != '']
        continue_cond = (lambda x: (x.split('.')[0] not in lines or x.endswith(".svg")))
        files = [0, 0]
        # lim = 50 if IS_WIN32 else 100000
        lim = 100000
        # print(join(self.folder_top, 'ShoeV2_sketch'))
        post_fix = ["jpg", "jpeg","png"]
        ds = 'ShoeV2' if exists(join(self.folder_top, 'ShoeV2_sketch')) else 'ChairV2'
        self.skim[SK], files[SK] = _load_all_ims(join(self.folder_top, '{}_sketch'.format(ds)), continue_cond=continue_cond, logger=self.logger,
                                                 ret_f=True, single_ch=True, lim=lim, post_fix=post_fix, edge=None)
        self.skim[IM], files[IM] = _load_all_ims(join(self.folder_top, '{}_photo'.format(ds)), continue_cond=continue_cond, logger=self.logger,
                                                 ret_f=True, lim=lim, post_fix=post_fix, edge=None)
        self.skim_paths = [list(files[SK]), list(files[IM])]
        def get_pref(p, what):
            pp = p
            p = os.path.splitext(os.path.split(p)[-1])[0]
            if what == SK:
                f = p.rfind('_')
                if f != -1: p = p[:f]
            return p
        for what in [SK, IM]:
            files[what] = [get_pref(f, what) for f in files[what]]
        correspond = [[[] for _ in range(len(files[SK]))], [[] for _ in range(len(files[IM]))]]
        for i, n in enumerate(files[SK]):
            correspond[IM][files[IM].index(n)].append(i)
            correspond[SK][i].append(files[IM].index(n))
        self.fg_label = [[], []]
        self.fg_label[IM] = np.arange(0, len(self.skim[IM]))
        self.fg_label[SK] = np.array([correspond[SK][i][0] for i in range(len(self.skim[SK]))])
        self.correspond = correspond
        self.print("correspond: {} {}".format(len(correspond[SK]), len(correspond[IM])))
        # 2490312525.
        for what in [SK, IM]:
            for i in range(len(correspond[what])):
                # print(what==SK,i, len(correspond[what][i]))
                if len(correspond[what][i]) == 0:
                    self.print("warning: {}-{}::{} has 0 correspond".format(what, i, self.skim_paths[what][i]))

    def num(self, what=IM):
        return len(self.skim[what])

    def __getitem__(self, idx):
        """
        :param index: index of the data
        :return: a tensor list [sketch, image]. Whether the returned object is sketch or image is decided by
            self.mode
        """
        idx_im = idx % self.num(IM)
        flip = idx_im // self.num(IM)
        im = self.skim[IM][idx_im]
        idx_sk = np.random.choice(self.correspond[IM][idx_im])
        sk = expand3(self.skim[SK][idx_sk])
        ret = self.return_data(sk, im, flip, idx_sk, idx_im)
        return ret

    def traverse(self, what, batch_size=16, skip=1):
        assert what == IM or what == SK, "CMT_dataloader.traverse: what must be IM({})/SK({}), but get {}"\
            .format(IM, SK, what)
        rets_ims = []; rets_ids = []; paths = []
        for i, x in enumerate(self.skim[what]):
            if i % skip != 0: continue
            x = expand3(x)
            paths.append(self.skim_paths[what][i])
            x = data_afg_ssl.trans_noaug(x)
            rets_ims.append(x)
            rets_ids.append(self.fg_label[what][i])
            if len(rets_ims) == batch_size:
                yield torch.stack(rets_ims, dim=0), torch.tensor(rets_ids), paths
                rets_ims = []; rets_ids = []; paths = []
        if len(rets_ids) >= 1:
            yield torch.stack(rets_ims, dim=0), torch.tensor(rets_ids), paths

    def __len__(self):
        # return len(self.fg_label[IM]) * self.n_flip if self.flip == 1 else len(self.fg_label[IM])
        return self.num(IM) * self.n_flip if self.flip == 1 else self.num(IM)


class ADRAM_dataloader_simple(ADRAM_dataloader_clean):
    def __init__(self, folder_top=None, logger=None, sz=None, train=True, aug=True, flip=0, vanilla=1):
        '''
        support Chairs/Shoes dataset
        '''
        super(ADRAM_dataloader_simple, self).__init__(
            folder_top=folder_top, logger=logger, sz=sz, train=train, aug=aug, flip=flip, vanilla=vanilla)
        train = 'train' if self.train else 'test'
        self.skim = [[], []]
        self.skim_paths = [[], []]
        post_fix = ["jpg", "jpeg", "png"]
        self.skim[SK], self.skim_paths[SK], self.skim[IM], self.skim_paths[IM] = \
            *_load_all_ims(join(folder_top, train, 'sketches'), post_fix=post_fix,  logger=self.logger,
                           ret_f=True, edge=None),\
            *_load_all_ims(join(folder_top, train, 'images'), post_fix=post_fix,  logger=self.logger,
                           ret_f=True, edge=None)
        # print(self.skim_paths[SK])
        self.print("sk: {},\t\tim: {}".format(len(self.skim[SK]), len(self.skim[IM])))

    def num(self, what=IM):
        return len(self.skim[what])

    def __getitem__(self, idx):
        """
        :param index: index of the data
        :return: a tensor list [sketch, image]. Whether the returned object is sketch or image is decided by
            self.mode
        """
        idx_im = idx % self.num(IM)
        flip = idx // self.num(IM)
        idx_sk = idx_im
        im = self.skim[IM][idx_im]
        sk = expand3(self.skim[SK][idx_sk])
        ret = self.return_data(sk, im, flip, idx_sk, idx_im)
        return ret

    def traverse_get_x_path_i(self, x, what, i):
        return x, self.skim_paths[what][i], i

    def __len__(self):
        # return len(self.fg_label[IM]) * self.n_flip if self.flip == 1 else len(self.fg_label[IM])
        return self.num() * self.n_flip if self.flip == 1 else self.num()

