from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
import torch.nn.functional as F


def _get_pre_from_matches(matches):
    """
    :param matches: A n-by-m matrix. n is number of test samples, m is the top m elements used for evaluation
    :return: precision
    """
    return np.mean(matches)


def _map_change(inputArr):
    dup = np.copy(inputArr)
    for idx in range(inputArr.shape[1]):
        if idx != 0:
            # dup cannot be bool type
            dup[:,idx] = dup[:,idx-1] + dup[:,idx]
    return np.multiply(dup, inputArr)


def _get_map_all(gt_labels_query, gt_labels_gallery, dists):
    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(gt_labels_query.shape[0]):
        inst_id, gt_labels = gt_labels_query[fi], gt_labels_gallery

        pos_flag = gt_labels == inst_id
        tot_pos = np.sum(pos_flag)

        sort_idx = np.argsort(dists[fi])
        tp = pos_flag[sort_idx]
        fp = np.logical_not(tp)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        try:
            rec = tp / tot_pos
            prec = tp / (tp + fp)
            mrec = np.append(0, rec)
            mrec = np.append(mrec, 1)

            mpre = np.append(0, prec)
            mpre = np.append(mpre, 0)

            for ii in range(len(mpre) - 2, -1, -1):
                mpre[ii] = max(mpre[ii], mpre[ii + 1])

            msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
            mapi = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
        except:
            # print(inst_id, tot_pos)
            mapi = np.nan
        mAP_ls[gt_labels_query[fi]].append(mapi)
    return np.mean(sum(mAP_ls, []))


def _get_map_from_matches(matches):
    """
    mAP's calculation refers to https://github.com/ShivaKrishnaM/ZS-SBIR/blob/master/trainCVAE_pre.py.
    :param matches: A n-by-m matrix. n is number of test samples, m is the top m elements used for evaluation
            matches[i][j] == 1 indicates the j-th retrieved test image j belongs to the same class as test sketch i,
            otherwise, matches[i][j] = 0.
    :return: mAP
    """
    temp = [np.arange(matches.shape[1]) for _ in range(matches.shape[0])]
    mAP_term = 1.0 / (np.stack(temp, axis=0) + 1.0)
    # print("_map_change(matches)=\n",_map_change(matches))
    # print("mAP_term=\n",mAP_term)
    precisions = np.multiply(_map_change(matches), mAP_term)
    mAP = np.mean(precisions, axis=1)
    return np.mean(mAP)


def _eval(feats_labels_sk, feats_labels_im, n=200, batch_size=None, ret_ind=False, indices=None,
          cosine=True, data_trans=lambda x:x.half(), model=None, A=None, ebs=None, inf=True, dist_f=None):
    # 250000000
    indices_ = indices
    len_sk = len(feats_labels_sk[0])
    if batch_size is None:
        batch_size = 3000*3000 // len(feats_labels_im[0])
    i = 0
    matches = np.zeros(shape=[len_sk, n], dtype=np.uint32)
    indices = np.zeros(shape=[len_sk, n], dtype=np.uint32)
    model_dist.im = None
    while i < len_sk:
        m = min(i + batch_size, len_sk)
        matches[i:m, :], indices[i:m, :] = _eval_single_batch(
            [feats_labels_sk[0][i:m, :], feats_labels_sk[1][i:m]], feats_labels_im, n=n, cosine=cosine, inf=inf,
            data_trans=data_trans, model=model, A=A, ebs=ebs, indices=None if indices_ is None else indices_[i:m],
            dist_f=dist_f)
        i = m
    pre = _get_pre_from_matches(matches[:, :n])
    mAP = _get_map_from_matches(matches[:, :n])
    if ret_ind:
        return pre, mAP, indices
    return pre, mAP


def _eval_multicomp(feats_sk_im1, feats_sk_im2, feats_sk_im3, label_sk, label_im, n=200, batch_size=None,
                    comps=None, data_trans=lambda x:x.half()):
    # feats_labels_paire1 = [feats_sk, feats_im]
    len_sk = len(label_sk)
    if batch_size is None:
        batch_size = 50000000 // len(label_im)
    gap = [0., .25, .5, .75, 1.]
    if comps is None:
        comps = []
        for g1 in gap:
            for g2 in gap:
                if g1 + g2 < 1.001:
                    g3 = 1.0 - g1 - g2
                    comps.append([g1, g2, g3])
    matches = np.zeros(shape=[len(comps), len_sk, n], dtype=np.uint8)
    ds_oris = np.zeros(shape=[len(comps), len_sk, len(label_im)], dtype=np.float32)
    i = 0
    while i < len_sk:
        m = min(i + batch_size, len_sk)
        # matches[i:m, :]
        matches_comps, ds_ori = _eval_single_batch_multicomp(
            [feats_sk_im1[0][i:m, :], feats_sk_im1[1]],
            [feats_sk_im2[0][i:m, :], feats_sk_im2[1]],
            [feats_sk_im3[0][i:m, :], feats_sk_im3[1]]
            , label_sk[i:m], label_im, n=n, comps=comps, data_trans=data_trans)
        for match, matches_curr, ds_ori, ds_ori_curr in zip(matches, matches_comps, ds_oris, ds_ori):
            # print(match[i:m, :].shape, matches_curr.shape)
            match[i:m, :] = matches_curr
            ds_ori[i:m, :] = ds_ori_curr
        i = m
    for comp, match, ds_ori in zip(comps, matches, ds_oris):
        comp.append([_get_pre_from_matches(match[:, :n].astype(np.uint32)), _get_map_from_matches(match[:, :n].astype(np.uint32)), _get_map_all(label_sk,label_im,ds_ori)])
    return comps




# cos d
def tc_dist_cosine(feature1, feature2, data_trans=lambda x: x.half()):
    feature1 = torch.from_numpy(feature1)
    feature1 = data_trans(feature1)
    feature2 = torch.from_numpy(feature2)
    feature2 = data_trans(feature2)
    feature1 = feature1.cuda()
    feature2 = feature2.cuda()
    feature1 = F.normalize(feature1)
    feature2 = F.normalize(feature2)
    distance = 2 - torch.matmul(feature1, feature2.T)
    return distance

# eul d
def tc_dist_v1(feature1, feature2, data_trans=lambda x: x.half()):
    feature1 = torch.from_numpy(feature1).cuda()
    feature2 = torch.from_numpy(feature2).cuda()
    a = feature1
    b = feature2
    a_l2 = torch.sum(a ** 2, -1)
    b_l2 = torch.sum(b ** 2, -1)
    ab = a @ b.T
    d = a_l2.reshape([a_l2.shape[0], 1]) + b_l2.reshape([1, b_l2.shape[0]]) - 2 * ab
    d = torch.sqrt(torch.relu(d) + 1e-10)
    return d

# eul d
def tc_dist_v2(feature1, feature2, data_trans=lambda x: x.half()):
    feature1 = torch.from_numpy(feature1)
    feature2 = torch.from_numpy(feature2)
    feature1 = feature1.cuda()
    feature2 = feature2.cuda()
    distance = torch.cdist(feature1, feature2)
    distance = distance
    return distance


tc_dist = tc_dist_v2


def model_dist(feature1, feature2, model, A=None, ebs=None, indices=None, inf=True):
    d1 = []
    feature1 = torch.from_numpy(feature1)
    feature2 = torch.from_numpy(feature2)
    # print(feature1.shape, feature2.shape)
    if A is None:
        A = [None, None]
    if ebs is None:
        ebs = [None, None]
    # print(111, feature1.shape)
    for i, f1 in enumerate(feature1):
        if model_dist.im is None:
            model_dist.im = feature2.cpu()
            im = model_dist.im
            A_1 = A[1]
            ebs_1 = ebs[1]
        elif indices is not None:
            # print(type(indices))
            im = model_dist.im[indices[i]]
            A_1 = A[1][indices[i]]
            ebs_1 = ebs[1][indices[i]]

        if A[0] is None:
            cls, sk, im = model(f1.cuda(), im, 1, None, None, inf=inf)
        else:
            # print(11111, ebs[0][i].shape, *ebs[0][i].shape)
            # print(ebs[0][i].reshape([1, *ebs[0][i].shape]).shape, A[0][i].reshape([1, *A[0][i].shape]).shape)
            cls, sk, im = model(f1.cuda(), im, 1, A[0][i].reshape([1, *A[0][i].shape]), A_1,
                                           ebs[0][i].reshape([1, *ebs[0][i].shape]), ebs_1, inf=inf)
        if model_dist.im is None:
            model_dist.im = im
        if indices is None:
            d1.append(cls)
        else:
            d1.append(cls[:indices.shape[1]])
    d1 = torch.stack(d1)
    model_dist.im = None
    # print(222222, d1.shape)
    return d1


model_dist.im = None


def _eval_single_batch_multicomp(feats_sk_im1, feats_sk_im2, feats_sk_im3, label_sk, label_im, n, comps, data_trans=lambda x: x.half()):
    """
    :param feats_labels_sk: a two-element tuple [features_of_sketches, labels_of_sketches]
        labels_of_sketches and labels_of_images are scalars(class id).
    :param feats_labels_im: a two-element tuple [features_of_images, labels_of_images]
            features_of_images and features_of_sketches are used for distance calculation.
    :param n: the top n elements used for evaluation
    :return: precision@n, mAP@n, mAP@all
    """
    ds1 = tc_dist_cosine(feats_sk_im1[0], feats_sk_im1[1])
    ds1 = data_trans(ds1).cpu().float()
    ds2 = tc_dist_cosine(feats_sk_im2[0], feats_sk_im2[1])
    ds2 = data_trans(ds2).cpu().float()
    ds3 = tc_dist_cosine(feats_sk_im3[0], feats_sk_im3[1])
    ds3 = data_trans(ds3).cpu().float()
    retrieved_classes = np.zeros(shape=[len(comps), len(ds3), n], dtype=np.uint32)
    dss = np.zeros(shape=[len(comps), *ds1.shape], dtype=np.float)
    for j, (c1, c2, c3) in enumerate(comps):
        ds = c1 * ds1 + c2 * ds2 + c3 * ds3
        dss[j,:,:] = ds
        indices = torch.argsort(ds.half().cuda(), 1)[:, :n].cpu().numpy()
        retrieved_classes[j, :, :] = label_im[indices]
    for i in range(retrieved_classes.shape[1]):
        retrieved_classes[:, i, :] = (retrieved_classes[:, i, :] == label_sk[i])
    return retrieved_classes, dss


def _eval_single_batch(feats_labels_sk, feats_labels_im, n=200, cosine=True, data_trans=lambda x: x.half(), model=None,
                       A=None, ebs=None, indices=None, inf=None, dist_f=None):
    """
    :param feats_labels_sk: a two-element tuple [features_of_sketches, labels_of_sketches]
        labels_of_sketches and labels_of_images are scalars(class id).
    :param feats_labels_im: a two-element tuple [features_of_images, labels_of_images]
            features_of_images and features_of_sketches are used for distance calculation.
    :param n: the top n elements used for evaluation
    :return: precision@n, mAP@n, mAP@all
    """
    # ds = cdist(feats_labels_sk[0], feats_labels_im[0], 'cosine')
    if inf is None:
        inf = n * 2
    if dist_f is not None:
        ds = model_dist(feats_labels_sk[0], feats_labels_im[0], model=model, A=A, ebs=ebs, indices=indices, inf=inf)
    else:
        if model is not None:
            ds = model_dist(feats_labels_sk[0], feats_labels_im[0], model=model, A=A, ebs=ebs, indices=indices, inf=inf)
        elif cosine:
            ds = tc_dist_cosine(feats_labels_sk[0], feats_labels_im[0], data_trans=data_trans)
        else:
            ds = tc_dist(feats_labels_sk[0], feats_labels_im[0], data_trans=data_trans)
    ds = data_trans(ds)
    # nn = NN(n_neighbors=len(feats_labels_im[0]), metric='cosine', algorithm='brute').fit(feats_labels_im[0])
    # _, indices = nn.kneighbors(feats_labels_sk[0])
    # print(ds)
    indices = torch.argsort(ds, 1)[:, :n]
    indices = indices.cpu().numpy()
    # print('_eval_ori', indices[0])
    retrieved_classes = np.array(feats_labels_im[1])[indices]
    matches = np.vstack([(retrieved_classes[i] == feats_labels_sk[1][i])
                       for i in range(retrieved_classes.shape[0])]).astype(np.uint32)
    # print(indices[0])
    # print(matches.shape, indices.shape, retrieved_classes.shape, np.array(feats_labels_im[1]).shape) # (201, 1) (201, 10) (201, 10, 300) (30, 300)
    return matches, indices


def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def cal_matrics(dists_cosine1, dists_cosine2, image_l, sketch_l, n=200):
    precision_list = list()
    mAP_list = list()
    lambda_list = list()
    for lambda_i in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
        dists = lambda_i*dists_cosine1 + (1-lambda_i)*dists_cosine2
        rank = np.argsort(dists, 0)
        ranksn = rank[:n, :].T
        classesn = np.array([[image_l[i] == sketch_l[r] for i in ranksn[r]] for r in range(len(ranksn))]) # ske_size*n
        precision = np.mean(classesn)
        mAP = np.mean(np.sum(classesn*np.cumsum(classesn, axis=1)/np.cumsum(np.ones(classesn.shape), axis=1), axis=1)/n)
        precision_list.append(precision)
        mAP_list.append(mAP)
        lambda_list.append(lambda_i)
    min_dists = np.minimum(dists_cosine1, dists_cosine2)
    rank = np.argsort(min_dists, 0)
    ranksn = rank[:n, :].T
    classesn = np.array([[image_l[i] == sketch_l[r] for i in ranksn[r]] for r in range(len(ranksn))]) # ske_size*n
    precision_c = np.mean(classesn)
    mAP_c = np.mean(np.sum(classesn*np.cumsum(classesn, axis=1)/np.cumsum(np.ones(classesn.shape), axis=1), axis=1)/n)
    return precision_list, mAP_list, lambda_list, precision_c, mAP_c


def cal_matrics_dist(dists, image_l, sketch_l, n=200):
    rank = np.argsort(dists, 0)
    ranksn = rank[:n, :].T
    classesn = np.array([[image_l[i] == sketch_l[r] for i in ranksn[r]] for r in range(len(ranksn))]) # ske_size*n
    precision = np.mean(classesn)
    mAP = np.mean(np.sum(classesn*np.cumsum(classesn, axis=1)/np.cumsum(np.ones(classesn.shape), axis=1), axis=1)/n)
    return precision, mAP


def cal_matrics_single(image_f, image_l, sketch_f, sketch_l, n=200):
    dists = cdist(image_f, sketch_f, 'cosine')
    rank = np.argsort(dists, 0)
    ranksn = rank[:n, :].T
    classesn = np.array([[image_l[i] == sketch_l[r] for i in ranksn[r]] for r in range(len(ranksn))]) # ske_size*n
    precision = np.mean(classesn)
    mAP = np.mean(np.sum(classesn*np.cumsum(classesn, axis=1)/np.cumsum(np.ones(classesn.shape), axis=1), axis=1)/n)
    return precision, mAP


eval_multicomp = _eval_multicomp
eval = _eval
get_map_all = _get_map_all


def __eval(feats_labels_sk, feats_labels_im, n=200, cosine=True):
    """
    :param feats_labels_sk: a two-element tuple [features_of_sketches, labels_of_sketches]
        labels_of_sketches and labels_of_images are scalars(class id).
    :param feats_labels_im: a two-element tuple [features_of_images, labels_of_images]
            features_of_images and features_of_sketches are used for distance calculation.
    :param n: the top n elements used for evaluation
    :return: precision@n, mAP@n, mAP@all
    """
    # nn = NN(n_neighbors=n, metric='euclidean', algorithm='brute').fit(feats_labels_im[0])
    if 1:
        nn = NN(n_neighbors=len(feats_labels_im[0]), metric='cosine' if cosine else 'euclidean', algorithm='brute').fit(feats_labels_im[0])
        _, indices = nn.kneighbors(feats_labels_sk[0])
    else:
        ds = cdist(feats_labels_sk[0], feats_labels_im[0])
        indices = np.argsort(ds, 1)

    retrieved_classes = np.array(feats_labels_im[1])[indices]
    matches = np.vstack([(retrieved_classes[i] == feats_labels_sk[1][i])
                       for i in range(retrieved_classes.shape[0])]).astype(np.uint16)
    return _get_pre_from_matches(matches[:, :n]), _get_map_from_matches(matches[:, :n])


def eval_np(feats_labels_sk, feats_labels_im, n=200, batch_size=None, ret_ind=False, indices=None,
          cosine=True, data_trans=lambda x:x.half(), model=None, A=None, ebs=None, inf=True, dist_f=None):
    return __eval(feats_labels_sk, feats_labels_im, n, cosine=cosine)


def get_map_all_cuda(gt_labels_query, pos_flags, trans=lambda x:x):
    tot_poss = torch.sum(pos_flags, 1)
    tps = torch.cumsum(pos_flags, dim=1)
    mAP_ls = [[] for _ in range(len(torch.unique(gt_labels_query)))]
    mrec = trans(torch.zeros(pos_flags.shape[1] + 2, dtype=torch.float).cuda())
    mrec[-1] = 1
    mpre = trans(torch.zeros(pos_flags.shape[1] + 2, dtype=torch.float).cuda())
    mpre[-1] = 0
    tppfp = trans(torch.arange(1, pos_flags.shape[1]+1).cuda().float())
    mrecs = trans(torch.zeros([gt_labels_query.shape[0], pos_flags.shape[1] + 2], dtype=torch.float).cuda())
    mrecs[:,-1] = 1
    mpres = trans(torch.zeros([gt_labels_query.shape[0], pos_flags.shape[1] + 2], dtype=torch.float).cuda())
    mpres[:,-1] = 0
    mpres[:,1:-1] = tps / tppfp.reshape([1,-1])
    mrecs[:,1:-1] = tps / tot_poss.reshape([-1, 1])
    for ii in range(mpres.shape[1] - 2, -1, -1):
        mpres[:,ii] = torch.max(torch.stack([mpres[:, ii], mpres[:, ii + 1]]), dim=0)[0]
    mrecs1 = mrecs[:,1:]
    mrecsm1 = mrecs[:,:-1]
    msks = torch.cat([pos_flags, torch.zeros([pos_flags.shape[0], 1], dtype=pos_flags.dtype).cuda()], -1).bool()
    mpres1 = mpres[:,1:]
    for fi in range(gt_labels_query.shape[0]):
        mski = msks[fi]
        mapi = torch.sum((mrecs1[fi][mski] - mrecsm1[fi][mski]) * mpres1[fi][mski]).cpu().numpy()
        mAP_ls[gt_labels_query[fi]].append(mapi)
    return np.mean(sum(mAP_ls, []))


def get_metrics_all_cuda(gt_labels_query, gt_labels_gallery, dists, ps=[100, 200], maps=[200], mapall=True):
    gt_labels_query_cuda = torch.from_numpy(gt_labels_query).cuda()
    gt_labels_gallery_cuda = torch.from_numpy(gt_labels_gallery).cuda()
    pos_flags = (gt_labels_query_cuda.reshape([-1, 1]) == gt_labels_gallery_cuda.reshape([1,-1])).float().half()
    del gt_labels_gallery_cuda

    d_sort_indices = torch.argsort(torch.from_numpy(dists).cuda().half())
    for fi in range(gt_labels_query.shape[0]):
        pos_flags[fi] = pos_flags[fi][d_sort_indices[fi]]
    if mapall:
        mapall = get_map_all_cuda(gt_labels_query_cuda.int(), pos_flags)
    del gt_labels_query_cuda
    del d_sort_indices
    ps = [torch.mean(pos_flags[:,:p]).cpu().item() for p in ps]
    maps = [_get_map_from_matches_cuda(pos_flags[:,:p]) for p in maps ]

    return mapall, ps, maps


def tc_dist_cosine_simple(feature1, feature2):
    feature1 = F.normalize(feature1)
    feature2 = F.normalize(feature2)
    distance = 2 - torch.matmul(feature1, feature2.T)
    return distance


def tc_dist_simple(query, gallery, cosine=True, trans=lambda x:x):
    if cosine:
        ds = tc_dist_cosine_simple(trans(query), trans(gallery))
    else:
        ds = torch.cdist(query.float(), gallery.float())
    return trans(ds)


# feat = np.random.rand(5,3)
# labels = np.random.permutation(5)
# _,_,_,_,indices = get_metrics_all_cuda_from_feat(labels,labels,feat,feat, ps=[1], maps=[], mapall=False)


def argsort_in_batch(dists, bs=None):
    if bs is None:
        bs = (dists.shape[0] // 10) if dists.shape[0] > 10 else dists.shape[0]
    i = 0
    d_sort_indices = torch.zeros(dists.shape, dtype=torch.int).cuda()
    while True:
        if i >= dists.shape[0]: break
        d_sort_indices[i:i+bs] = torch.argsort(dists[i:i+bs])
        i += bs
    return d_sort_indices

def get_metrics_all_cuda_from_feat(gt_labels_query, gt_labels_gallery, feat_query, feat_gallery, ps=[100, 200],
                                   maps=[200], cos=True, half=1, mapall=True, dists=None):
    if half == 0:
        trans = lambda x:x
    elif half == 1:
        trans = lambda x: x.half()
    elif half == -1:
        trans = lambda x: x.double()
    else:
        raise Exception("Half should be -1, 0, or 1")
    if dists is None:
        if isinstance(feat_query, np.ndarray):
            feat_query = torch.from_numpy(feat_query)
            feat_gallery = torch.from_numpy(feat_gallery)
        dists = trans(tc_dist_simple(trans(feat_query.cuda()), trans(feat_gallery.cuda()), cos, trans))
    d_sort_indices = argsort_in_batch(dists)
    del dists

    if isinstance(gt_labels_query, np.ndarray):
        gt_labels_query = torch.from_numpy(gt_labels_query)
        gt_labels_gallery = torch.from_numpy(gt_labels_gallery)
    gt_labels_query_cuda = gt_labels_query.cuda()
    gt_labels_gallery_cuda = gt_labels_gallery.cuda()
    pos_flags = trans((gt_labels_query_cuda.reshape([-1, 1]) == gt_labels_gallery_cuda.reshape([1,-1])).float())
    del gt_labels_gallery_cuda

    # d_sort_indices = d_sort_indices.long()
    for fi in range(gt_labels_query.shape[0]):
        pos_flags[fi] = pos_flags[fi][d_sort_indices[fi].long()]
    d_sort_indices = d_sort_indices.cpu()

    if mapall: mapall = get_map_all_cuda(gt_labels_query_cuda, pos_flags, trans)
    del gt_labels_query_cuda

    ps = [torch.mean(pos_flags[:,:p]).cpu().item() for p in ps]
    maps = [_get_map_from_matches_cuda(pos_flags[:,:p], trans) for p in maps ]
    pos_flags = pos_flags.cpu()
    return mapall, ps, maps, pos_flags, d_sort_indices


def _get_map_from_matches_cuda(matches, trans=lambda x:x):
    """
    mAP's calculation refers to https://github.com/ShivaKrishnaM/ZS-SBIR/blob/master/trainCVAE_pre.py.
    :param matches: A n-by-m matrix. n is number of test samples, m is the top m elements used for evaluation
            matches[i][j] == 1 indicates the j-th retrieved test image j belongs to the same class as test sketch i,
            otherwise, matches[i][j] = 0.
    :return: mAP
    """
    # temp = [np.arange(matches.shape[1]) for _ in range(matches.shape[0])]
    matches = trans((torch.from_numpy(matches).cuda() if isinstance(matches, np.ndarray) else matches.cuda()).float())
    mAP_term = (1.0 / trans(torch.arange(1, 1 + matches.shape[1]).cuda().float())).unsqueeze(0)
    dup = torch.cumsum(matches, dim=1)
    precisions = dup * matches * mAP_term
    mAP = torch.mean(precisions, 1)
    return torch.mean(mAP).cpu().item()


def _test_eval_multicomp():
    # feats_sk_im1, feats_sk_im2, feats_sk_im3, label_sk, label_im, n=200, batch_size=None, comps=None
    """
    [0.0, 0.0, 1.0, [0.33433911159263274, 0.11784650297832018]]
    [0.0, 0.25, 0.75, [0.33432286023835317, 0.11747442926786932]]
    """
    np.random.seed(0)
    l_sk = 923
    l_im = 533
    d = 2
    feats_sk_im1 = [np.random.rand(l_sk, d), np.random.rand(l_im, d)]
    d += 0
    feats_sk_im2 = [np.random.rand(l_sk, d), np.random.rand(l_im, d)]
    d += 0
    feats_sk_im3 = [np.random.rand(l_sk, d), np.random.rand(l_im, d)]
    label_sk = np.random.randint(0, 3, [l_sk])
    label_im = np.random.randint(0, 3, [l_im])
    ret = _eval_multicomp(feats_sk_im1, feats_sk_im2, feats_sk_im3, label_sk, label_im, batch_size=100)
    print(__eval([feats_sk_im1[0], label_sk], [feats_sk_im1[1], label_im]))
    print('111')
    for r in ret:
        print(r)


def _test_eval():
    # feats_sk_im1, feats_sk_im2, feats_sk_im3, label_sk, label_im, n=200, batch_size=None, comps=None
    l_sk = 923
    l_im = 533
    d = 2
    feats_sk_im1 = [np.random.rand(l_sk, d), np.random.rand(l_im, d)]
    label_sk = np.random.randint(0, 3, [l_sk])
    label_im = np.random.randint(0, 3, [l_im])
    label_sk[5:5+200] = 1
    label_im[30:30+200] = 1
    feats_sk_im1[0][5:5 + 200] *= 0.1
    feats_sk_im1[1][30:30 + 200] *= 0.1
    ret = _eval([feats_sk_im1[0], label_sk], [feats_sk_im1[1], label_im], n=200, batch_size=100, ret_ind=0, cosine=True)
    for r in ret:
        print(r)
    print('\n\n')
    ret = __eval([feats_sk_im1[0], label_sk], [feats_sk_im1[1], label_im], n=200)
    for r in ret:
        print(r)


def _test__get_map_from_matches():
    a = np.array([
        #[1,1,1,1,1],
        [0,1,0,1,1],
        # [1,1,1,1,0]
    ])
    print(a)
    print(_get_map_from_matches(a))


if __name__=='__main__':
    _test__get_map_from_matches()
    # _test_eval()

