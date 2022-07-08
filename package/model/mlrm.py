import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torch.optim import Adam
from package.model.utils import num_params
import torch.nn.functional as F
import math


class TripletLoss(nn.Module):
    """
    The common triplet loss
    """
    def __init__(self):
        super(TripletLoss, self).__init__()

    def calc_with_d(self, pos_distance, negative_distance, m=0.3):
        triplet_loss = pos_distance - negative_distance + m  # bs x bs x num_vec
        eye = torch.eye(pos_distance.shape[0]).to(pos_distance.device).bool()
        triplet_loss = F.relu(triplet_loss[~eye])
        return triplet_loss.mean()

    def forward(self, sketch, photo, m=0.3):
        """
        :param sketch: [bs, dim]
        :param photo: [bs, dim]
        :param m: margin
        :return: triplet loss
        """
        pos_distance = torch.norm(sketch - photo, p=2, dim=1)
        negative_distance = torch.norm(sketch.unsqueeze(0) - photo.unsqueeze(1), p=2, dim=2)
        return self.calc_with_d(pos_distance, negative_distance, m)


import copy
class TransformerEncoder(nn.Module):
    """
    A common transformer encoder
    """
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # self multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # fead forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x2 = self.norm1(x)
        x2 = self.self_attn(x2, x2, x2)[0]
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)
        return x


class RegionLevelAttention(nn.Module):
    def __init__(self, tf_dim, dims, levels, regions, num_layers=3, l2norm=0):
        super(RegionLevelAttention, self).__init__()
        self.tf_dim = tf_dim
        self.l2norm = l2norm
        self.dims = dims
        self.regions = regions
        self.levels = levels

        # E^G, it encodes geometry maps into positional encodings
        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(levels, self.tf_dim // 4, kernel_size=3, stride=2),
            nn.BatchNorm2d(self.tf_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.tf_dim // 4, self.tf_dim, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        # it reduces multi-level region features into the same dimension
        self.feat_reducers = nn.ModuleList([
            nn.Sequential(nn.Linear(d, tf_dim)) for d in dims])
        encoder_layer = TransformerEncoderLayer(tf_dim, 4, tf_dim)
        self.transen = TransformerEncoder(encoder_layer, num_layers)

        # two fc heads after the transformer encoder
        self.level_weight_head = nn.Sequential(
            nn.Linear(tf_dim, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.region_weight_head = nn.Sequential(
            nn.Linear(tf_dim, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.Linear(32, 1))

    def _obtain_feat_each_level(self, maps_w, low_featmaps, global_rn=False):
        bs = maps_w.shape[0]
        n_levels = len(low_featmaps)
        feats_each_level = []
        for fm in low_featmaps:
            fm = fm[:,None,...].repeat(1, maps_w.shape[1], 1, 1, 1)
            feats_level = (fm * maps_w).mean(-1).mean(-1)
            if global_rn:
                # to construct a fake negative image by replacing a region with the region from another image
                neg_i = np.random.randint(0, feats_level.shape[1])
                feats_level = torch.cat([feats_level[:,:neg_i],
                                         feats_level.roll(1, 0)[:, neg_i:neg_i+1],
                                         feats_level[:,neg_i+1:]], 1)
            feats_each_level.append(feats_level)
        feats_each_level_ = [self.feat_reducers[i](feats_each_level[i].reshape([-1, self.dims[i]])).reshape([bs, -1, self.tf_dim])
                            for i in range(n_levels)]
        feats_each_level_.append(sum(feats_each_level_) / len(feats_each_level_))
        if self.l2norm:
            feats_each_level = [F.normalize(f, p=2, dim=-1) for f in feats_each_level]
        return feats_each_level, torch.cat(feats_each_level_, 1)

    def _obtain_positions(self, maps, n_levels):
        bs = maps.shape[0]
        maps_level = torch.zeros(bs, (n_levels + 1) * self.regions, n_levels, *maps.shape[-2:]).to(maps.device)

        # construct geometry map for each region feature
        for r in range(self.regions):
            for l in range(n_levels):
                maps_level[:, r + l * self.regions, l] = maps[:, r]
            # Level l+1 geometry map
            maps_level[:, r + n_levels * self.regions, :] = maps[:, r, None, ...]

        # use geometry encoder to obtain positional embeddings
        return self.geometry_encoder(maps_level.reshape(-1, *maps_level.shape[2:])).\
            reshape([bs, (n_levels + 1) * self.regions, -1])

    def forward(self, feats, maps, global_rn=False):
        bs = maps.shape[0]
        n_levels = len(feats)
        feats_each_level_raw, feats_each_level = self._obtain_feat_each_level(maps[:,:,None], feats, global_rn)
        # feats_each_level: l1r1, l1r2, l1r3, l2r1, l2r2, l2r3, ...

        # obtain positional embeddings
        position_embedding = self._obtain_positions(maps.detach(), n_levels)

        # embedded reduced features
        feats_each_level_enc = self.transen(position_embedding + feats_each_level)

        # obtain level weights W_L
        level_weight = self.level_weight_head(feats_each_level_enc[:,:-self.regions].reshape([-1, self.tf_dim])).reshape([bs,-1])
        level_weight = torch.softmax(level_weight.reshape([bs, -1, self.regions]), -1)
        level_weight = level_weight.permute(0,2,1)

        # obtain region weights W_N
        region_weight = self.region_weight_head(feats_each_level_enc[:,-self.regions:].reshape([-1, self.tf_dim])).reshape([bs,-1])
        region_weight = torch.softmax(region_weight, -1)
        return feats_each_level_raw, region_weight, level_weight, position_embedding


class AttentionCNN(nn.Module):
    """
    Spacial attention module, inspired by CBAM
    """
    def __init__(self, levels, atts, lama_sig, kernel_size=7):
        super(AttentionCNN, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.atts = atts
        self.lama_sig = lama_sig
        self.conv1 = nn.Sequential(
            nn.Conv2d(levels * 2, 16, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.atts, kernel_size, padding=padding, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_levels):
        avg_outs = [torch.mean(x, dim=1, keepdim=True) for x in x_levels]
        max_outs = [torch.max(x, dim=1, keepdim=True)[0] for x in x_levels]
        x = torch.cat(avg_outs + max_outs, dim=1)
        x = self.conv1(x)
        act_thresh = x.reshape([*x.shape[:2], -1]).sort(-1)[0][:,:,-self.lama_sig]
        return torch.sigmoid(x - act_thresh[:,:,None,None]) if self.lama_sig > 0 else torch.sigmoid(x)


class DRE(nn.Module):
    """
    Feature extractor based on DenseNet169
    """
    def __init__(self, args):
        super(DRE, self).__init__()
        self.args = args
        self.mk_bb()
        assert self.args.only_inter_layer == 0
        if args.atts > 1:
            self.condition_encoder = nn.Sequential(
                nn.Linear(args.atts, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1280),
            )
        self.spatial_attention = AttentionCNN(len(self.low_dims), 1, self.args.lama_sig)
        self.indicator_vector = torch.eye(self.args.atts).unsqueeze(0)

    def mk_bb(self):
        self.net = models.densenet169(True, memory_efficient=self.args.cp)
        self.low_dims = [256, 512,1280,1664][self.args.level_start:]

        # reduce feature maps sizes in low level
        self.reduction_nets = nn.ModuleList([self.cnn_reduce(512), self.cnn_reduce(1280)])

    def cnn_reduce(self, ch):
        if ch == 256: raise Exception("wrong reduction size")
        if ch == 512:
            return nn.Sequential(
                nn.Conv2d(ch, ch // 4, 3, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 4, ch, 3, 2, 1),
                nn.ReLU(inplace=True))
        if ch == 1280: # sz = 16
            return nn.Sequential(
                nn.Conv2d(ch, ch // 4, 3, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 4, ch, 1, 1, 0),
                nn.ReLU(inplace=True))

    def forward(self, x):
        low_featmaps = []
        x = self.net.features[:5](x)
        low_featmaps.append(x)
        x = self.net.features[5:7](x)
        low_featmaps.append(self.reduction_nets[0](x))
        x = self.net.features[7:9](x)
        x_map_gen = x
        low_featmaps.append(self.reduction_nets[1](x))
        low_featmaps = low_featmaps[self.args.level_start:]
        x = self.net.features[9:](x)
        spatial_attentions = []
        if x.is_cuda: self.indicator_vector = self.indicator_vector.cuda()

        # LAMA way to extract multiple regions
        for i in range(self.args.atts):
            cond_conv = self.condition_encoder(self.indicator_vector[:,i])[:,:,None,None]
            x_map = x_map_gen + cond_conv
            x_map = self.net.features[9:](x_map)
            spatial_attentions.append(self.spatial_attention(low_featmaps + [x_map]))
        spatial_attentions = torch.cat(spatial_attentions, 1)
        low_featmaps.append(x)
        return low_featmaps, spatial_attentions


class OverlappingLoss(nn.Module):
    def __init__(self, ovl_type):
        super(OverlappingLoss, self).__init__()
        assert ovl_type in [0,1]
        self.ovl_type = ovl_type

    def cama_loss(self, maps):
        # CAMA overlapping penalty
        mult = maps[:, 0, ...]
        for i in range(1, maps.shape[1]):
            mult = mult * maps[:, i, ...]
        loss = mult.sum(-1).sum(-1).mean() / maps.shape[1]
        return loss

    def lama_loss(self, maps):
        # LAMA overlapping penalty
        forward_maxpool = []
        backward_maxpool = []
        for pools, maps_ in [[forward_maxpool, maps],
                             [backward_maxpool, maps.flip(1)]]:
            pools.append(maps_[:, 0])
            for i in range(1, maps.shape[1]):
                pools.append(torch.maximum(pools[-1], maps_[:, i]))
        loss = 0
        atts = maps.shape[1]
        for i in range(atts):
            if i == 0:
                maximum_others = backward_maxpool[-2]
            elif i == atts - 1:
                maximum_others = forward_maxpool[-2]
            else:
                maximum_others = torch.maximum(forward_maxpool[i - 1], backward_maxpool[atts - i - 2])
            loss = loss + (maximum_others * maps[:, i]).sum(-1).sum(-1).mean()
        return loss / (maps.shape[1] * (maps.shape[1] - 1))

    def forward(self, maps):
        if maps.shape[1] == 1:
            return 0
        if self.ovl_type == 0:
            return self.cama_loss(maps)
        else:
            return self.lama_loss(maps)


EPS = 1e-5


class MLRM(nn.Module):
    ITEM_NAMES = {
        'loss_local_triplet': 1.0,
        'loss_sk_ovl': 1.0,
        'loss_im_ovl': 1.0,
        'loss_global_triplet': 1.0,
        'loss_param_alpha': 0.1,
        'loss_param_beta': 0.1,
    }
    def bb2sizes(bb):
        if bb == 'gn':
            return {"in": 224}
        elif bb == 'ic3':
            return {"in": 299}
        elif bb[:2] == 'dn':
            return {"in": 256}
        else:
            return {"in": 224}
    def __init__(self,   args, logger=None):
        super(MLRM,  self).__init__()
        # single 5e-5 att5 oil1  43.8
        self.logger = logger
        self.args = args
        self.adj_step = 1
        assert self.args.bb == 'dn169', "Bad _init_net bb option: {}".format(self.args.bb)
        self.nets = nn.ModuleList([DRE(args=self.args)]) if self.args.single \
            else nn.ModuleList([DRE(args=self.args), DRE(args=self.args)])
        self.params = [self.nets]
        self.triplet = TripletLoss()
        self.overlap = OverlappingLoss(ovl_type=self.args.ovl)
        self.rla = RegionLevelAttention(self.args.tf_dim, self.net.low_dims, len(self.net.low_dims), self.args.atts, l2norm=args.l2norm)
        self.params.append(self.rla)
        Opt = Adam
        self.opt = Opt(sum([list(m.parameters()) for m in self.params], []), lr=args.lr * 0.1)
        self.weights = args.weights if isinstance(args.weights, dict) else eval(args.weights)

        for s in MLRM.ITEM_NAMES:
            if s not in self.weights:
                self.weights[s] = MLRM.ITEM_NAMES[s]

        self.print("\nMLRM:\nnum_params: {}\topt_params: {} \ninput: {} \nweights: {} \n".format(
            num_params(self), len(list(self.parameters())), self.args.weights, self.weights))

    @property
    def net(self):
        return self.nets[0]

    @property
    def net_sk(self):
        return self.nets[0]

    @property
    def net_im(self):
        return self.nets[0] if self.args.single else self.nets[1]

    def print(self,  s):
        if self.logger is None:
            print(s)
        else:
            self.logger.info('{}'.format(s))

    def forward(self, net, x):
        feats, maps = net(x)
        feats_each_level, reg_weights, lvl_weights, _ = self.rla(feats, maps)
        if self.args.base_debug:
            feats_each_level[-1] = feats[-1].mean(-1).mean(-1)[:,None,:]
        return {"feats_each_level": feats_each_level,
                "reg_weights": reg_weights,
                "lvl_weights": lvl_weights,
               "maps": maps}

    def forward_sk(self,  x):
        return self.forward(self.net_sk, x)

    def forward_im(self,  x):
        return self.forward(self.net_im, x)

    def adjust_learning_rate(self):
        self.adj_step += 1
        if self.adj_step % 1000 == 0 and self.adj_step > self.args.warmup:
            lr = self.args.lr * math.pow(self.args.decay, float(self.adj_step) / self.args.max_step)
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr
                self.print("step={} lr={}".format(self.adj_step, param_group['lr']))

    def _calc_local_trp(self, sk, im):
        """
        input: b sketches, b images
        output: local triplet loss
        """
        trp = 0
        if self.args.individual_trp == 0:
            for i, (f_sk, f_im) in enumerate(zip(sk, im)):
                for j in range(self.args.atts):
                    trp = trp + self.triplet(f_sk[:,j,:], f_im[:,j,:], m=self.args.trp)
        if self.args.individual_trp == 1:
            """
            We derive local triplet loss here, including instance-negative and region-negative triplet losses 
            """
            bs = sk[0].shape[0]; n_regions = self.args.atts
            mat_margin = torch.eye(bs).to(sk[0].device).repeat_interleave(n_regions,-2).repeat_interleave(n_regions,-1)
            close_margin_w = self.weights['loss_param_alpha']
            # we prepare a margin matrix
            mat_margin = (1 - mat_margin) * self.args.trp * (1 - close_margin_w) + self.args.trp * close_margin_w
            for i, (f_sk, f_im) in enumerate(zip(sk, im)):
                """
                f_sk: (b x N) x dim
                f_im: (b x N) x dim
                mat_margin: (b x N) x (b x N), 
                            each diagonal block of N x N size serves for region-negative triplets
                            other elements serves for instance-negative triplet s
                """
                level_adjust_coeff = 1 + math.e ** (-i -1)
                sk_individual = f_sk.reshape([-1, self.net.low_dims[i]])
                im_individual = f_im.reshape([-1, self.net.low_dims[i]])
                trp = trp + self.triplet(sk_individual, im_individual, m=mat_margin * level_adjust_coeff)
        return trp

    def _ovl_loss(self, maps):
        maps = torch.clamp(maps, min=self.args.mapmin)
        return self.overlap(maps)

    def _get_feats(self, sk, im):
        bs = sk.shape[0]
        if self.args.single:
            feats_skim, maps_skim = self.net(torch.cat([sk, im], 0))
            feats_sk = [f[:bs] for f in feats_skim]; feats_im = [f[bs:] for f in feats_skim]
            maps_sk = maps_skim[:bs]; maps_im = maps_skim[bs:]
        else:
            feats_sk, maps_sk = self.net_sk(sk)
            feats_im, maps_im = self.net_im(im)
            feats_sk = self.postprocess_feats(feats_sk, maps_sk)
            feats_im = self.postprocess_feats(feats_im, maps_im)
        return feats_sk, feats_im, maps_sk, maps_im

    def _calc_22dist(self, feats_each_level_raw_sk, feats_each_level_raw_im, region_weight_sk,
                            region_weight_im, level_weight_sk, level_weight_im):
        """
        input: m sketches, n images
        output: a m-by-n L2 distance matrix
        """
        n_levels = len(feats_each_level_raw_sk)
        region_weight_unpaired = (region_weight_sk[:,None,:] + region_weight_im[None, :,:]) * 0.5
        neg_d = []
        for l in range(n_levels):
            level_weight_unpaired = (level_weight_sk[:,None,:,l] + level_weight_im[None,:,:,l]) * 0.5 * region_weight_unpaired
            neg_d.append(torch.norm(feats_each_level_raw_sk[l][:, None, ...] -
                                    feats_each_level_raw_im[l][None, ...], p=2, dim=-1) * level_weight_unpaired)
        neg_d = (torch.stack(neg_d, -1) * region_weight_unpaired[:,:,:,None]).sum(-1).sum(-1)
        return neg_d

    def _calc_weighted_loss(self, feats_each_level_raw_sk, feats_each_level_raw_im, region_weight_sk,
                            region_weight_im, level_weight_sk, level_weight_im):
        """
        input: n sketches, n images. The i-th image and the i-th sketch are paired.
        output: triplet loss
        """
        trp = self.args.trp
        neg_d = self._calc_22dist(feats_each_level_raw_sk, feats_each_level_raw_im, region_weight_sk,
                                         region_weight_im, level_weight_sk, level_weight_im)
        bs = level_weight_sk.shape[0]
        eye = torch.eye(bs).to(neg_d.device).bool()
        triplet_loss = neg_d[eye][:,None] - neg_d + trp
        triplet_loss = F.relu(triplet_loss[~eye])
        global_loss = triplet_loss.mean()
        return eye, neg_d[eye][:,None], global_loss

    def _calc_global_trp(self,
            feats_each_level_raw_sk, feats_each_level_raw_im, feats_each_level_raw_im_rn,
            region_weight_sk, region_weight_im, region_weight_im_rn,
            level_weight_sk, level_weight_im, level_weight_im_rn):
        """
        input: n sketches, n images, n constructed images.
        output: global triplet loss
        """
        # gtrp_in: global instance-negative triplet loss
        eye, postive_d, gtrp_in = self._calc_weighted_loss(feats_each_level_raw_sk, feats_each_level_raw_im, region_weight_sk,
                            region_weight_im, level_weight_sk, level_weight_im)

        # gtrp_rn: global region-negative triplet loss
        trp = self.weights['loss_param_beta'] * self.args.trp
        neg_d = self._calc_22dist(feats_each_level_raw_sk, feats_each_level_raw_im_rn, region_weight_sk,
                                         region_weight_im_rn, level_weight_sk, level_weight_im_rn)
        gtrp_rn = postive_d - neg_d + trp
        gtrp_rn = F.relu(gtrp_rn[~eye]).mean()
        return gtrp_in + gtrp_rn

    def get_loss(self, sk, im):
        losses = {}

        # obtain feature maps and attention maps
        feats_sk, feats_im, maps_sk, maps_im = self._get_feats(sk, im)

        # obtain retrieval features and attention weights
        feats_each_level_raw_sk, region_weight_sk, level_weight_sk, _ = \
            self.rla(feats_sk, maps_sk)
        feats_each_level_raw_im, region_weight_im, level_weight_im, _ = \
            self.rla(feats_im, maps_im)

        # obtain the constucted image feature set
        feats_each_level_raw_im_rn, region_weight_im_rn, level_weight_im_rn, _ = \
            self.rla(feats_im, maps_im, global_rn=True)

        losses['loss_global_triplet'] = self._calc_global_trp(
            feats_each_level_raw_sk, feats_each_level_raw_im, feats_each_level_raw_im_rn,
            region_weight_sk, region_weight_im, region_weight_im_rn,
            level_weight_sk, level_weight_im, level_weight_im_rn) \
            if self.weights['loss_global_triplet'] > EPS else 0
        losses['loss_sk_ovl'] = self._ovl_loss(maps_sk) if self.weights['loss_sk_ovl'] > EPS else 0
        losses['loss_im_ovl'] = self._ovl_loss(maps_im) if self.weights['loss_im_ovl'] > EPS else 0
        losses['loss_local_triplet'] = self._calc_local_trp(feats_each_level_raw_sk, feats_each_level_raw_im) \
            if self.weights['loss_local_triplet'] > EPS else 0
        loss_total = sum([losses[k] * self.weights[k] for k in losses])
        losses['loss_total'] = loss_total
        return losses

    def optimize_params(self,  sk,  im, idx_sk, idx_im):
        losses = self.get_loss(sk, im)
        self.adjust_learning_rate()
        self.opt.zero_grad()
        losses['loss_total'].backward()
        self.opt.step()
        for k in losses: losses[k] = float(losses[k])
        losses = dict([(k, losses[k]) for k in losses if abs(losses[k]) > EPS])
        return losses