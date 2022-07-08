import numpy as np
import torch

from package.args.mlrm_args import parse_config
import os
args = parse_config()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from package import get_metrics_all_cuda_from_feat
from torch.utils.data import DataLoader
from package.model.mlrm import *
from package.dataset.data_mlrm import *
import time
from package.model.utils import *

F = nn.functional
IS_WIN32 = 1

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


def mean_loss_sum(loss_sum):
    return [round(np.mean(loss),5) for loss in loss_sum]


# feats = {"labels": [], "feats_each_level": [], "reg_weights": [], "lvl_weights": [], "maps": [], "paths": []}
def save_visualization(feats_labels_sk, feats_labels_im, sk2im_idx, folder, sk_num=10, target_n=10, base_size=224):
    feats_labels_sk = [feats_labels_sk['labels'].numpy(), feats_labels_sk['maps'].numpy(), feats_labels_sk['paths']]
    feats_labels_im = [feats_labels_im['labels'].numpy(), feats_labels_im['maps'].numpy(), feats_labels_im['paths']]
    target_n = min(target_n, sk2im_idx.shape[1])
    feats_labels_sk = [item[:sk_num] for item in feats_labels_sk]
    sk2im_idx = sk2im_idx[:sk_num, :target_n]
    split = os.path.split
    get_name = lambda fn: split(fn)[-1]
    get_cls_name = lambda fn: split(split(fn)[0])[-1]
    def visual_a_row(sk_fn, im_fns,
                     sk_map, im_maps, hits,
                     clr_sk=(255,0,0), clr_hit=(0,255,0), clr_miss=(0,0,255)):
        clrs = [clr_sk] + [[clr_miss, clr_hit][int(h)] for h in hits]
        fns = [sk_fn] + im_fns
        sk_ims = [cv2.resize(cv2.imread(fn), (base_size, base_size)) for fn in fns]
        visual_ims = [
            cv2.copyMakeBorder(sk_im, 8,8,8,8,cv2.BORDER_CONSTANT, value=clr)
            for sk_im, clr in zip(sk_ims, clrs)]
        rows = [np.concatenate(visual_ims, 1)]
        for i in range(sk_map.shape[0]):
            hotmap_visuals = [visual_hotmap(sk_map[i], sk_ims[0])] + \
                             [visual_hotmap(im_maps[i_t][i], sk_ims[i_t+1]) for i_t in range(target_n)]
            visual_ims_i = [
                cv2.copyMakeBorder(sk_im, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=clr)
                for sk_im, clr in zip(hotmap_visuals, clrs)]
            rows.append(np.concatenate(visual_ims_i, 1))
        return np.concatenate(rows, 0)
    for i in range(sk_num):
        fn_retrieval = join(mkdir(folder), "retrieval_{}_{}___{}".format(
            i, get_cls_name(feats_labels_sk[-1][i]),
            get_name(feats_labels_sk[-1][i]).replace(".png", ".jpg")))

        visualization_row = visual_a_row(
            feats_labels_sk[-1][i], [feats_labels_im[-1][i_t] for i_t in sk2im_idx[i]],
            feats_labels_sk[-2][i], [feats_labels_im[-2][i_t] for i_t in sk2im_idx[i]],
            [feats_labels_im[0][i_t] == feats_labels_sk[0][i] for i_t in sk2im_idx[i]],
        )
        cv2.imwrite(fn_retrieval, visualization_row)


def to_vanilla_form(feats, crops=10):
    def arr2vanilla(arr):
        if isinstance(arr, list):
            arr = torch.Tensor(arr)
        return arr.reshape([-1, crops, *arr.shape[1:]]).transpose(0,1)
    for k in feats:
        if k == "paths": continue
        if isinstance(feats[k], list):
            for i in range(len(feats[k])):
                feats[k][i] = arr2vanilla(feats[k][i])
        else:
            feats[k] = arr2vanilla(feats[k])
    return feats


def _get_test_metrics(args, model, in_feats_sk, in_feats_im, ps, mtype='standard'):
    assert mtype.startswith('standard') and mtype.split('standard')[1] == 'l2'
    crops = 10 - args.vanilla * 9
    in_feats_sk = to_vanilla_form(in_feats_sk, crops=crops)
    in_feats_im = to_vanilla_form(in_feats_im, crops=crops)
    if mtype.startswith('standard'):
        bs = args.batch_size
        d_alls = []
        for i in range(crops):
            def get_cropi(xk):
                if isinstance(xk, list):
                    return [xk_[i] for xk_ in xk]
                else:
                    return xk[i]
            feats_im = dict([(k, get_cropi(in_feats_im[k])) for k in in_feats_im if k != "paths"])
            feats_sk = dict([(k, get_cropi(in_feats_sk[k])) for k in in_feats_sk if k != "paths"])
            sk_num = feats_sk["reg_weights"].shape[0]
            im_num = feats_im["reg_weights"].shape[0]
            d_all = torch.zeros(sk_num, im_num).cuda()
            feats_sk["feats_each_level"] = [f.cuda() for f in feats_sk["feats_each_level"]]
            feats_sk["reg_weights"] = feats_sk["reg_weights"].cuda()
            feats_sk["lvl_weights"] = feats_sk["lvl_weights"].cuda()
            im_bs = bs * 5000 // sk_num
            for i_im in range(0, im_num, im_bs):
                d_all[:, i_im: i_im + im_bs] =\
                    model._calc_22dist(feats_sk["feats_each_level"],
                                       [f[i_im: i_im + im_bs].cuda() for f in feats_im["feats_each_level"]],
                                       feats_sk["reg_weights"],
                                       feats_im["reg_weights"][i_im: i_im + im_bs].cuda(),
                                        feats_sk["lvl_weights"].cuda(),
                                       feats_im["lvl_weights"][i_im: i_im + im_bs].cuda())
            d_alls.append(d_all)
        d_all = torch.mean(torch.stack(d_alls), 0)
        _, accs, _, _, sk2im_idx = get_metrics_all_cuda_from_feat(
            in_feats_sk["labels"][0], in_feats_im["labels"][0], None, None, ps=ps, maps=[], mapall=False, half=False, dists=d_all)
    else:
        assert args.vanilla == 1
        if mtype.startswith('single_feat'):
            fi = int(mtype[-1])
            f_sk = feats_sk["feats_each_level"][-fi][:,0]
            f_im = feats_im["feats_each_level"][-fi][:,0]
        elif mtype.startswith('sum_last'):
            fi = int(mtype[-1])
            f_sk = feats_sk["feats_each_level"][-fi].mean(-2)
            f_im = feats_im["feats_each_level"][-fi].mean(-2)
        else:
            raise Exception("Error mtype: {}".format(mtype))
        _, accs, _, _, sk2im_idx = get_metrics_all_cuda_from_feat(
            feats_sk["labels"], feats_im["labels"], f_sk, f_im, ps=ps, maps=[], mapall=False, half=False)
    for i, (acc, ni) in enumerate(zip(accs, ps)):  accs[i] = round(acc * ni, 3)
    return accs, sk2im_idx


def get_weight_stat(feats):
    def cat_mean_std(mat):
        return torch.cat([mat.mean(0), mat.std(0)], -1)
    norm2 = lambda x: F.normalize(x, p=2, dim=-1)
    dl_weights = [torch.norm(norm2(x)[:,None,...] - norm2(x)[:,:,None,...],
                             p=2, dim=-1).mean(0).sum() / (x.shape[1] ** 2 - x.shape[1])
                 for x in feats["feats_each_level"]]
    dl_weights2 = [torch.norm(x[:,None,...] - x[:,:,None,...],
                             p=2, dim=-1).mean(0).sum() / (x.shape[1] ** 2 - x.shape[1])
                 for x in feats["feats_each_level"]]
    return cat_mean_std(feats["reg_weights"]), cat_mean_std(feats["lvl_weights"]), dl_weights, dl_weights2


def _test_and_save(epochs, dataloader_test, model, logger, args, loss_sum, f_pref="", save=True, steps=0, losst=1):
    if not hasattr(_test_and_save, 'best_acc'):
        _test_and_save.best_acc = -100000
        _test_and_save.last_acc = -100000
    logger.info("\n------------------------------------------------TESTING----------------------------------------------------")
    def e(n, skip=1, pref=""):
        if not isinstance(n, list): n = [n]
        start_cpu_t = time.time()
        accs0 = 0
        with torch.no_grad():
            model.eval()
            feats_sk, feats_im = _extract_feats_sk_im(
                data=dataloader_test.dataset, model=model, batch_size=args.batch_size, skip=skip)
            feat_t = time.time()
            for mtype in ['standardl2']:
                accs, _test_and_save.sk2im_idx = _get_test_metrics(args, model, feats_sk, feats_im, ps=n, mtype=mtype)
                accs0 = max(accs0, accs[0])
                eval_t = time.time()
                logger.info(str(mtype) + " " + pref + "  acc@{}: {}, skip: {}".format(n, accs, skip))
            logger.info(("{} data:{}-{}.\nepochs: {}, bestPre: {:.3G}, (cpu time: {:.3G}/{:.3G}/{:.3G}s)\nloss: {}, ").format( f_pref,
                  [f.shape for f in feats_sk["feats_each_level"]], [f.shape for f in feats_im["feats_each_level"]],
                  epochs, _test_and_save.best_acc, eval_t - start_cpu_t, feat_t - start_cpu_t, eval_t - feat_t, loss2string(loss_sum,False)))
        return accs0
    pre = e([1,5,10], pref=" ")
    _test_and_save.last_acc = pre
    if pre >= _test_and_save.best_acc and save:
        _test_and_save.best_acc = pre
    d = {'model': model.state_dict(), 'epochs': epochs, 'steps': steps, 'args': args}
    if args.save:
        torch.save(d, save_fn(args.save_dir, epochs, pre))
    torch.cuda.empty_cache()


def save_fn(save_dir, it, pre=0):
    return join(mkdir(join(save_dir, 'models')), 'Iter__{}__{}.pkl'.format(it, int(pre * 1000)))


def _try_load(args, logger, model):
    if args.start_from is None:
        files = os.listdir(mkdir(join(mkdir(args.save_dir), 'models')))
        if len(files) == 0:
            logger.info("Cannot find any checkpoint. Start new training.")
            return 0, 0
        latest = max(files, key=lambda name: int(os.path.split(name)[-1].split('.')[0].split('__')[1]))
        checkpoint = join(args.save_dir, 'models', latest)
    else:
        try: checkpoint = save_fn(args.save_dir, str(int(args.start_from)))
        except: checkpoint = args.start_from if exists(args.start_from) else\
            join(mkdir(join(args.save_dir, 'models')), args.start_from)

    ckpt = torch.load(checkpoint, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['model'].items()
                       if (k in model_dict and model_dict[k].shape == v.shape)}  # filter out unnecessary keys
    model_dict.update(pretrained_dict)
    d_lens = [len(pretrained_dict), len(model_dict), len(ckpt['model'])]
    logger.info("_init_model from {}.  pretrained_dict/model_dict/ckpt params: {}/{}/{}".
                format(checkpoint, *d_lens))
    if not d_lens[0] == d_lens[1] == d_lens[2]:
        logger.info("WARNING: d_lens: {} unequal!".format(d_lens))
    model.load_state_dict(model_dict, strict=False)
    return ckpt['epochs'], ckpt['steps'] if 'steps' in ckpt else 0


def _extract_feats_sk_im(data, model, batch_size=64, skip=1):
    feats_labels_sk = _extract_feats(data, model.forward_sk, SK, skip=skip,
                                     batch_size=batch_size)
    feats_labels_im = _extract_feats(data, model.forward_im, IM, skip=skip,
                                     batch_size=batch_size)
    return feats_labels_sk, feats_labels_im


def _extract_feats(data_test, model, what, skip=1, batch_size=16):
    feats = {"labels": [], "feats_each_level": [], "reg_weights": [], "lvl_weights": [], "maps": [], "paths": []}
    for batch_idx, (xs, id, path) in \
            enumerate(data_test.traverse(what, skip=skip, batch_size=batch_size)):
        outs_b = model(xs.cuda())
        for k in ["lvl_weights", "maps", "reg_weights"]: outs_b[k] = outs_b[k].cpu()
        outs_b['feats_each_level'] = [f.cpu() for f in outs_b['feats_each_level']]
        for k in outs_b: feats[k].append(outs_b[k])
        feats['labels'].append(id)
        feats['paths'].append(list(path))
    feats_each_level_all = []
    for i in range(len(feats["feats_each_level"][-1])):
        feats_each_level_all.append(torch.cat([f[i] for f in feats["feats_each_level"]], 0))
    feats["feats_each_level"] = feats_each_level_all
    for k in ["reg_weights", "lvl_weights", "maps"]:
        feats[k] = torch.cat(feats[k], 0)
    feats['labels'] = torch.from_numpy(np.concatenate(feats['labels'], 0))
    feats['paths'] = sum(feats['paths'], [])
    return feats


from package.dataset import data_afg_ssl


def _init_dataset(args, logger):
    sizes = MLRM.bb2sizes(args.bb)
    args.aug_params = eval(args.aug_params)
    data_afg_ssl.make_globals(sz=int(sizes['in'] * args.aug_params['crop']), crop_size=sizes['in'], aug_params=args.aug_params)
    if args.dataset == 'sketchy':
        data_train = ADRAM_dataloader_Sketchy(folder_top=args.folder_top, logger=logger, train=True,  aug=args.aug,skip=args.skip, flip=args.aug_params['flip'])
        data_test = ADRAM_dataloader_Sketchy(folder_top=args.folder_top, logger=logger, train=False,  aug=False, skip=args.skip, vanilla=args.vanilla)
    elif args.dataset in ['shoes2', 'chairs2']:
        data_train = ADRAM_dataloader_multi(folder_top=args.folder_top, logger=logger, train=True,  aug=args.aug, flip=args.aug_params['flip'])
        data_test = ADRAM_dataloader_multi(folder_top=args.folder_top, logger=logger, train=False,  aug=False, vanilla=args.vanilla)
    else:
        data_train = ADRAM_dataloader_simple(folder_top=args.folder_top, logger=logger, train=True,  aug=args.aug, flip=args.aug_params['flip'])
        data_test = ADRAM_dataloader_simple(folder_top=args.folder_top, logger=logger, train=False,  aug=False, vanilla=args.vanilla)
    dataloader_train = DataLoader(dataset=data_train, batch_size=args.batch_size,
                                  num_workers=0 if IS_WIN32 else args.workers, shuffle=args.dataset != 'sketchy',
                                  drop_last=False)
    dataloader_test = DataLoader(dataset=data_test, batch_size=args.batch_size * 2, num_workers=0, shuffle=False,
                                 drop_last=False)
    logger.info("Datasets initialized")
    return args, dataloader_train, dataloader_test, logger


def optimize_params(model, batch):
    for i in range(len(batch)):
        batch[i] = batch[i].cuda()
    sk_ori, im_ori, idx_sk, idx_im = batch
    losses = model.optimize_params(sk_ori, im_ori, idx_sk, idx_im)
    del sk_ori, im_ori, idx_sk, idx_im, batch
    torch.cuda.empty_cache()
    return losses


def loss2string(losses, clear=True):
    ret = "{"
    for k in losses:
        ret += "{}: {:.3G}, ".format(k, np.mean(losses[k]))
        if clear: losses[k] = [losses[k][-1]]
    return ret + '}'


def _train(args, dataloader_train, dataloader_test, logger):
    model = MLRM(args=args,  logger=logger)
    chceck_params_rec(model, 3)
    model = model.cuda()
    epochs, step = _try_load(args, logger, model)
    logger.info(str(args))
    args.epochs += epochs
    model.train()
    loss_sum = {}

    def ts(f_pref, data_t, save=True, losst=1):
        _test_and_save(epochs=epochs, dataloader_test=data_t, f_pref=f_pref, save=save,
                           model=model, logger=logger, args=args, loss_sum=loss_sum, steps=step, losst=losst)

    ts(f_pref='data_test', data_t=dataloader_test, save=True)
    while True:
        for batch in dataloader_train:
            step += 1
            if batch[0].shape[0] != args.batch_size: continue

            losses = optimize_params(model, batch)
            for k in losses:
                if k in loss_sum: loss_sum[k].append(losses[k])
                else: loss_sum[k] = []

            if step % args.save_every == 0:
                ts(f_pref='data_test', data_t=dataloader_test, save=True)

            if step % args.print_every == 0:
                logger.info('epochs: {}, step: {}/{},\nloss: {}'.
                            format(epochs, step, args.max_step, loss2string(loss_sum)))
        dataloader_train.dataset.epoch()
        if step >= args.max_step: break
        epochs += 1


def train(args, logger):
    args, dataloader_train, dataloader_test, logger = _init_dataset(args=args, logger=logger)
    _train(args=args, dataloader_test=dataloader_test, dataloader_train=dataloader_train, logger=logger)


if __name__ == '__main__':
    logger = make_logger(join(mkdir(args.save_dir), curr_time_str() + '.log'))
    train(args, logger)

