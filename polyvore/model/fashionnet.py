"""Models for fashion Net."""
import logging
import threading
import time

import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import polyvore.config as cfg
import polyvore.param
import utils
from polyvore import debugger

from . import backbones as B
from . import basemodel as M

NAMED_MODELS = utils.get_named_function(B)
LOGGER = logging.getLogger(__name__)
OUTFIT_ENCODER = False


def contrastive_loss(margin, im, s):
    size, dim = im.shape
    scores = im.matmul(s.t()) / dim
    diag = scores.diag()
    zeros = torch.zeros_like(scores)
    # shape #item x #item
    # sum along the row to get the VSE loss from each image
    cost_im = torch.max(zeros, margin - diag.view(-1, 1) + scores)
    # sum along the column to get the VSE loss from each sentence
    cost_s = torch.max(zeros, margin - diag.view(1, -1) + scores)
    # to fit parallel, only compute the average for each item
    vse_loss = cost_im.sum(dim=1) + cost_s.sum(dim=0) - 2 * margin
    # for data parallel, reshape to (size, 1)
    return vse_loss / (size - 1)


def soft_margin_loss(x):
    target = torch.ones_like(x)
    return F.soft_margin_loss(x, target, reduction="none")


@utils.singleton
class RankMetric(threading.Thread):
    def __init__(self, num_users, args=(), kwargs=None):
        from queue import Queue

        threading.Thread.__init__(self, args=(), kwargs=None)
        self.daemon = True
        self.queue = Queue()
        self.deamon = True
        self.num_users = num_users
        self._scores = [[[] for _ in range(self.num_users)] for _ in range(4)]

    def reset(self):
        self._scores = [[[] for _ in range(self.num_users)] for _ in range(4)]

    def put(self, data):
        self.queue.put(data)

    def process(self, data):
        with threading.Lock():
            uidx, scores = data
            for u in uidx:
                for n, score in enumerate(scores):
                    for s in score:
                        self._scores[n][u].append(s)

    def run(self):
        print(threading.currentThread().getName(), "RankMetric")
        while True:
            data = self.queue.get()
            if data is None:  # If you send `None`, the thread will exit.
                return
            self.process(data)

    def rank(self):
        auc = utils.metrics.calc_AUC(self._scores[0], self._scores[1])
        binary_auc = utils.metrics.calc_AUC(self._scores[2], self._scores[3])
        ndcg = utils.metrics.calc_NDCG(self._scores[0], self._scores[1])
        binary_ndcg = utils.metrics.calc_NDCG(self._scores[2], self._scores[3])
        return dict(auc=auc, binary_auc=binary_auc, ndcg=ndcg, binary_ndcg=binary_ndcg)


class FashionNet(nn.Module):
    """Base class for fashion net."""

    def __init__(self, param):
        """See NetParam for details."""
        super().__init__()
        self.param = param
        self.scale = 1.0
        # user embedding layer
        self.user_embedding = M.UserEncoder(param)
        # feature extractor
        if self.param.use_visual:
            self.features = NAMED_MODELS[param.backbone](tailed=True)
        # single encoder or multi-encoders
        num_encoder = 1 if param.single else cfg.NumCate # NumCate = 3
        # hashing codes
        if self.param.use_visual:
            feat_dim = self.features.dim
            self.encoder_v = nn.ModuleList(
                [M.ImgEncoder(feat_dim, param) for _ in range(num_encoder)]
            )
            if OUTFIT_ENCODER:
                self.encoder_v.append(M.ImgEncoder(feat_dim * num_encoder, param))
        if self.param.use_semantic:
            feat_dim = 2400
            self.encoder_t = nn.ModuleList(
                [M.TxtEncoder(feat_dim, param) for _ in range(num_encoder)]
            )
            if OUTFIT_ENCODER:
                self.encoder_t.append(M.TxtEncoder(feat_dim * num_encoder, param))
        # matching block
        if self.param.hash_types == polyvore.param.NO_WEIGHTED_HASH:
            # use learnable scale
            self.core = M.LearnableScale(1)
        elif self.param.hash_types == polyvore.param.WEIGHTED_HASH_BOTH:
            # two weighted hashing for both user-item and item-item
            self.core = nn.ModuleList([M.CoreMat(param.dim), M.CoreMat(param.dim)])
        else:
            # single weighted hashing for user-item or item-item
            self.core = M.CoreMat(param.dim)
        if self.param.use_semantic and self.param.use_visual:
            self.loss_weight = dict(rank_loss=1.0, binary_loss=None, vse_loss=0.1)
        else:
            self.loss_weight = dict(rank_loss=1.0, binary_loss=None)
        self.configure_trace()
        self.rank_metric = RankMetric(self.param.num_users)
        if not self.rank_metric.is_alive():
            self.rank_metric.start()

    def gather(self, results):
        losses, accuracy = results
        loss = 0.0
        gathered_loss = {}
        for name, value in losses.items():
            weight = self.loss_weight[name]
            value = value.mean()
            # save the scale
            gathered_loss[name] = value.item()
            if weight:
                loss += value * weight
        # save overall loss
        gathered_loss["loss"] = loss.item()
        gathered_accuracy = {k: v.sum().item() / v.numel() for k, v in accuracy.items()}
        return gathered_loss, gathered_accuracy

    def get_user_binary_code(self, device="cpu"):
        uidx = np.arange(self.param.num_users).reshape(-1, 1)
        uidx = torch.from_numpy(uidx).to(device)
        one_hot = utils.one_hot(uidx, self.param.num_users)
        user = self.user_embedding(one_hot)
        return self.sign(user).cpu().numpy()

    def get_matching_weight(self):
        if self.param.hash_types == polyvore.param.WEIGHTED_HASH_BOTH:
            weights_u = self.core[0].weight.data.cpu().numpy()
            weights_i = self.core[1].weight.data.cpu().numpy()
            w = 1.0
        elif self.param.hash_types == polyvore.param.WEIGHTED_HASH_I:
            weights_u = []
            weights_i = self.core.weight.data.cpu().numpy()
            w = 1.0
        elif self.param.hash_types == polyvore.param.WEIGHTED_HASH_U:
            weights_u = self.core.weight.data.cpu().numpy()
            weights_i = []
            w = 1.0
        else:
            weights_u = []
            weights_i = []
            w = self.core.weight.data.cpu().numpy()
        return weights_i, weights_u, w

    def register_figure(self, tracer, title):
        for key, trace_dict in self.tracer.items():
            tracer.register_figure(
                title=title, xlabel="iteration", ylabel=key, trace_dict=trace_dict
            )

    def configure_trace(self):
        self.tracer = dict()
        self.tracer["loss"] = {
            "train.loss": "Train Loss(*)",
            "train.binary_loss": "Train Loss",
            "test.loss": "Test Loss(*)",
            "test.binary_loss": "Test Loss",
        }
        if self.param.use_semantic and self.param.use_visual:
            # in this case, the overall loss dose not equal to rank loss
            vse = {
                "train.vse_loss": "Train VSE Loss",
                "test.vse_loss": "Test VSE Loss",
                "train.rank_loss": "Train Rank Loss",
                "test.rank_loss": "Test Rank Loss",
            }
            self.tracer["loss"].update(vse)
        self.tracer["accuracy"] = {
            "train.accuracy": "Train(*)",
            "train.binary_accuracy": "Train",
            "test.accuracy": "Test(*)",
            "test.binary_accuracy": "Test",
        }
        self.tracer["rank"] = {
            "test.auc": "AUC(*)",
            "test.binary_auc": "AUC",
            "test.ndcg": "NDCG(*)",
            "test.binary_ndcg": "NDCG",
        }

    def __repr__(self):
        return super().__repr__() + "\n" + self.param.__repr__()

    def set_scale(self, value):
        """Set scale of TanH layer."""
        if not self.param.scale_tanh:
            return
        self.scale = value
        LOGGER.info("Set the scale to %.3f", value)
        self.user_embedding.set_scale(value)
        if self.param.use_visual:
            for encoder in self.encoder_v:
                encoder.set_scale(value)
        if self.param.use_semantic:
            for encoder in self.encoder_t:
                encoder.set_scale(value)

    def scores(self, ulatent, ilatents, scale=10.0):
        """For simplicity, we remove top-top pair on variable-length outfit and
           use duplicated top when necessary.
        """
        size = len(ilatents)
        indx, indy = np.triu_indices(size, k=1) # 上三角矩阵(不包括对角) indices
        if size == 4:
            # remove top-top comparison
            indx = indx[1:]
            indy = indy[1:]
        # N x size x D
        latents = torch.stack(ilatents, dim=1)
        ulatent = ulatent.unsqueeze(1).expand(-1, size, -1)
        # # N x size x D
        x = latents * ulatent
        # N x Comb x D
        y = latents[:, indx] * latents[:, indy]
        # shape [N]
        if self.param.hash_types == polyvore.param.WEIGHTED_HASH_BOTH:
            score_u = self.core[0](x).mean(dim=(1, 2))
            score_i = self.core[1](y).mean(dim=(1, 2))
        elif self.param.hash_types == polyvore.param.WEIGHTED_HASH_U:
            score_u = self.core(x).mean(dim=(1, 2))
            score_i = y.mean(dim=(1, 2))
        elif self.param.hash_types == polyvore.param.WEIGHTED_HASH_I:
            score_u = x.mean(dim=(1, 2))
            score_i = self.core(x).mean(dim=(1, 2))
        elif self.param.hash_types == polyvore.param.NO_WEIGHTED_HASH:
            score_u = self.core(x.mean(dim=(1, 2)))
            score_i = y.mean(dim=(1, 2))
        else:
            score_u = x.mean(dim=(1, 2))
            score_i = y.mean(dim=(1, 2))
        # scale score to [-2 * scale, 2 * scale]
        if self.param.zero_iterm:
            score = score_u * (scale * 2.0)
        elif self.param.zero_uterm:
            score = score_i * (scale * 2.0)
        else:
            score = (score_u + score_i) * scale
        # shape N x 1
        return score.view(-1, 1)

    def sign(self, x):
        """Return hash code of x.

        if x is {1,-1} binary, then h(x) = sign(x)
        if x is {1,0} binary, then h(x) = (sign(x - 0.5) + 1)/2
        """
        if self.param.binary01:
            return ((x.detach() - 0.5).sign() + 1) / 2.0
        return x.detach().sign()

    def latent_code(self, items, encoder):
        """
        Return latent codes.
        return encoder[cate](items)
        """
        latent_code = []
        size = len(items)
        if self.param.single:
            cate = [0] * size
        else:
            cate = self.param.cate_map
        latent_code = [encoder[c](x) for c, x in zip(cate, items)]
        # shape Length * N x D
        if OUTFIT_ENCODER:
            latent_code.append(encoder[-1](torch.cat(items, 1)))
        # shape Length * (N + 1) x D
        return latent_code

    def _pairwise_output(self, lcus, pos_feat, neg_feat, encoder):
        """return (pscore, nscore, bpscore, bnscore), latents"""
        lcpi = self.latent_code(pos_feat, encoder)
        lcni = self.latent_code(neg_feat, encoder)
        # score with relaxed features
        pscore = self.scores(lcus, lcpi)
        nscore = self.scores(lcus, lcni)
        # score with binary codes
        bcus = self.sign(lcus)
        bcpi = [self.sign(h) for h in lcpi]
        bcni = [self.sign(h) for h in lcni]
        bpscore = self.scores(bcus, bcpi)
        bnscore = self.scores(bcus, bcni)
        # stack latent codes
        # (N x Num) x D
        if self.param.variable_length:
            # only use second top item since to balance top category
            latents = torch.stack(lcpi[1:] + lcni[1:], dim=1)
        else:
            latents = torch.stack(lcpi + lcni, dim=1)
        latents = latents.view(-1, self.param.dim)
        return (pscore, nscore, bpscore, bnscore), latents

    def semantic_output(self, lcus, pos_feat, neg_feat):
        scores, latents = self._pairwise_output(
            lcus, pos_feat, neg_feat, self.encoder_t
        )
        debugger.put("item.s", latents)
        return scores, latents

    def visual_output(self, lcus, pos_img, neg_img):
        # extract visual features
        pos_feat = [self.features(x) for x in pos_img]
        neg_feat = [self.features(x) for x in neg_img]
        scores, latents = self._pairwise_output(
            lcus, pos_feat, neg_feat, self.encoder_v
        )
        debugger.put("item.v", latents)
        return scores, latents

    def debug(self):
        LOGGER.debug("Scale value: %.3f", self.scale)
        debugger.log("user")
        if self.param.use_visual:
            debugger.log("item.v")
        if self.param.use_semantic:
            debugger.log("item.s")
        if isinstance(self.core, nn.ModuleList):
            for module in self.core:
                module.debug()
        else:
            self.code.debug()

    def forward(self, *inputs):
        """Forward according to setting."""
        # pair-wise output
        posi, nega, uidx = inputs
        one_hot = utils.one_hot(uidx, self.param.num_users)
        # score latent codes
        lcus = self.user_embedding(one_hot)
        debugger.put("user", lcus)
        loss = dict()
        accuracy = dict()
        if self.param.use_semantic and self.param.use_visual:
            pos_v, pos_s = posi
            neg_v, neg_s = nega
            score_v, latent_v = self.visual_output(lcus, pos_v, neg_v)
            score_s, latent_s = self.semantic_output(lcus, pos_s, neg_s)
            scores = [0.5 * (v + s) for v, s in zip(score_v, score_s)]
            # visual-semantic similarity
            vse_loss = contrastive_loss(self.param.margin, latent_v, latent_s)
            loss.update(vse_loss=vse_loss)
        elif self.param.use_visual:
            scores, _ = self.visual_output(lcus, posi, nega)
        elif self.param.use_semantic:
            scores, _ = self.semantic_output(lcus, posi, nega)
        else:
            raise ValueError
        data = (uidx.tolist(), [s.tolist() for s in scores])
        self.rank_metric.put(data)
        diff = scores[0] - scores[1]
        binary_diff = scores[2] - scores[3]
        rank_loss = soft_margin_loss(diff)
        binary_loss = soft_margin_loss(binary_diff)
        acc = torch.gt(diff.data, 0)
        binary_acc = torch.gt(binary_diff.data, 0)
        loss.update(rank_loss=rank_loss, binary_loss=binary_loss)
        accuracy.update(accuracy=acc, binary_accuracy=binary_acc)
        return loss, accuracy

    def num_gropus(self):
        """Size of sub-modules."""
        return len(self._modules)

    def active_all_param(self):
        """Active all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_all_param(self):
        """Active all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_user_param(self):
        """Freeze user's latent codes."""
        self.active_all_param()
        for param in self.user_embedding.parameters():
            param.requires_grad = False

    def freeze_item_param(self):
        """Freeze item's latent codes."""
        self.freeze_all_param()
        for param in self.user_embedding.parameters():
            param.requires_grad = True

    def init_weights(self):
        """Initialize net weights with pre-trained model.

        Each sub-module should has its own same methods.
        """
        for model in self.children():
            if isinstance(model, nn.ModuleList):
                for m in model:
                    m.init_weights()
            else:
                model.init_weights()


class FashionNetDeploy(FashionNet):
    """Fashion Net.

    Fashion Net has three parts:
    1. features: Feature extractor for all items.
    2. encoder: Learn latent code for each category.
    3. user embedding: Map user to latent codes.
    """

    def __init__(self, param):
        """Initialize FashionNet.

        Parameters:
        num: number of users.
        dim: Dimension for user latent code.

        """
        super().__init__(param)

    def binary_v(self, items):
        feat = [self.features(x) for x in items]
        latent = self.latent_code(feat, self.encoder_v)
        binary_codes = [self.sign(h) for h in latent]
        return binary_codes

    def binary_t(self, items):
        latent = self.latent_code(items, self.encoder_t)
        binary_codes = [self.sign(h) for h in latent]
        return binary_codes

    def _single_output(self, lcus, feat, encoder):
        lcpi = self.latent_code(feat, encoder)
        # score with relaxed features
        score = self.scores(lcus, lcpi)
        # score with binary codes
        bcus = self.sign(lcus)
        bcpi = [self.sign(h) for h in lcpi]
        b_score = self.scores(bcus, bcpi)
        return score, b_score

    def semantic_output(self, lcus, feat):
        return self._single_output(lcus, feat, self.encoder_t)

    def visual_output(self, lcus, img):
        # extract visual features
        feat = [self.features(x) for x in img]
        return self._single_output(lcus, feat, self.encoder_v)

    def forward(self, *inputs):
        """Forward.

        Return the scores for items.
        """
        items, uidx = inputs
        one_hot = utils.one_hot(uidx, self.param.num_users)
        # compute latent codes
        users = self.user_embedding(one_hot)
        if self.param.use_semantic and self.param.use_visual:
            score_v = self.visual_output(users, items[0])
            score_s = self.semantic_output(users, items[1])
            score = [0.5 * (v + s) for v, s in zip(score_v, score_s)]
        elif self.param.use_visual:
            score = self.visual_output(users, items)
        elif self.param.use_semantic:
            score = self.semantic_output(users, items)
        else:
            raise ValueError
        return score

    def compute_codes(self, items):
        """Forward.

        Return the scores for items.
        """
        # compute latent codes
        if self.param.use_semantic and self.param.use_visual:
            items_v = self.binary_v(items[0])
            items_t = self.binary_t(items[1])
            return items_v, items_t
        elif self.param.use_visual:
            items_v = self.binary_v(items)
            return items_v
        elif self.param.use_semantic:
            items_t = self.binary_t(items)
            return items_t
        else:
            raise ValueError


class ColdStart(FashionNetDeploy):
    def __init__(self, param):
        """Initialize FashionNet.

        Parameters:

        """
        super().__init__(param)
        self.big_loader = polyvore.data.get_dataloader(param.big_data_param)
        self.small_loader = polyvore.data.get_dataloader(param.small_data_param)
        self.device = self.param.gpus[0]
        self.beta: float = 0.5
        self.big_outfits_posi_feat   = None
        self.big_outfits_nega_feat   = None
        self.small_outfits_posi_feat = None
        self.small_outfits_nega_feat = None
        self.big_user_codes          = None
        self.load_data()
        self.user_codes: dict = self.get_user_embedding()
    
    def get_outfits(self, loader):
        # outfits: user_num * outfit_num * 3(category) * 2(visual, text) * feat_dim
        # -------> user_num * feat_dim
        num_users = loader.num_users
        D = self.param.dim
        outfits = [torch.zeros(D) for _ in range(num_users)]
        outfits_num = [0 for _ in range(num_users)]
        pbar = tqdm.tqdm(loader, desc="Getting Outfits")
        for inputv in pbar:
            items, uidx = inputv
            items = utils.to_device(items, self.device)
            with torch.no_grad():
                feat_v, feat_t = self.compute_codes(items)
            feat_v = [feat.cpu().numpy().astype(np.int8) for feat in feat_v]
            feat_t = [feat.cpu().numpy().astype(np.int8) for feat in feat_t]
            for n, u in enumerate(uidx):
                # outfit = [[v[n], t[n]] for v, t in zip(feat_v, feat_t)]
                # outfits[u].append(outfit)
                outfit_v = sum([v[n] for v in feat_v]) / len(feat_v)
                outfit_t = sum([t[n] for t in feat_t]) / len(feat_t)
                outfits[u] += (outfit_v + outfit_t) / 2
                outfits_num[u] += 1
        # calculate average 
        outfits = [outfit / num for outfit, num in zip(outfits, outfits_num)]
        return outfits

    def get_user_binary_code(self):
        LOGGER.info('Loading big user_binary_code')
        with open('features/fashion_hash_net_vse_t3_u630.npz', 'rb') as f:
            data = pickle.load(f)
        return data['user_codes']

    def save_data(self):
        LOGGER.info('saving data')
        with open(self.param.data_file + '_new', 'wb') as f:
            data = dict(
                big_outfits_posi_feat   = self.big_outfits_posi_feat,
                big_outfits_nega_feat   = self.big_outfits_nega_feat,
                small_outfits_posi_feat = self.small_outfits_posi_feat,
                small_outfits_nega_feat = self.small_outfits_nega_feat,
                big_user_codes          = self.big_user_codes,
            )
            pickle.dump(data, f)
        LOGGER.info('save data done.')

    def load_data(self):
        self.cuda(self.device)
        try:
            with open(self.param.data_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            LOGGER.info(f'loading data failed: {e}')
        else:
            LOGGER.info(f'loading data from {self.param.data_file}')
            self.big_outfits_posi_feat   = data.get('big_outfits_posi_feat', None)
            self.big_outfits_nega_feat   = data.get('big_outfits_nega_feat', None)
            self.small_outfits_posi_feat = data.get('small_outfits_posi_feat', None)
            self.small_outfits_nega_feat = data.get('small_outfits_nega_feat', None)
            self.big_user_codes          = data.get('big_user_codes', None)

        if self.big_outfits_posi_feat is None:
            LOGGER.info('loading big_outfits_posi_feat')
            self.big_loader.set_data_mode('PosiOnly')
            self.big_outfits_posi_feat = self.get_outfits(self.big_loader)
        if self.big_outfits_nega_feat is None:
            LOGGER.info('loading big_outfits_nega_feat')
            self.big_loader.set_data_mode('NegaOnly')
            self.big_outfits_nega_feat = self.get_outfits(self.big_loader)
        reload_small = False
        if reload_small or self.small_outfits_posi_feat is None:
            LOGGER.info('loading small_outfits_posi_feat')
            self.small_loader.set_data_mode('PosiOnly')
            self.small_outfits_posi_feat = self.get_outfits(self.small_loader)
        if reload_small or self.small_outfits_nega_feat is None:
            LOGGER.info('loading small_outfits_nega_feat')
            self.small_loader.set_data_mode('NegaOnly')
            self.small_outfits_nega_feat = self.get_outfits(self.small_loader)
        if self.big_user_codes is None:
            LOGGER.info('loading big_user_codes')
            self.big_user_codes = self.get_user_binary_code()
        self.save_data()

    def user_user_embedding(self, big_outfits, small_outfits, big_user_codes, lambda_i):
        K = 300
        D = self.param.dim
        big_num_users = len(big_outfits)
        small_num_users = len(small_outfits)
        user_code = [torch.zeros(D) for _ in range(small_num_users)]
        pbar = tqdm.tqdm(range(small_num_users), desc="user_user_embedding")
        for ui in pbar:
            sim = [0.0 for uj in range(big_num_users)]
            for uj in range(big_num_users):
                sim[uj] = (small_outfits[ui] * lambda_i * big_outfits[uj]).sum() / D
            sim = sorted(enumerate(sim), key=lambda x:x[1])
            k = min(K, big_num_users)
            for uj, r in sim[:k]:
                user_code[ui] += r * big_user_codes[uj]
            user_code[ui] /= k
        return user_code

    def user_item_embedding(self, outfits, lambda_u):
        user_code = [outfit * lambda_u for outfit in outfits]
        return user_code

    def get_user_embedding(self):
        time_points = [time.time()]
        lambda_i, lambda_u, alpha = self.get_matching_weight()
        time_points.append(time.time())
        LOGGER.info(f'user_embedding() get weight time cost: {time_points[-1] - time_points[-2]}s.')
        codes_user_posi = self.user_user_embedding(
            self.big_outfits_posi_feat,
            self.small_outfits_posi_feat,
            self.big_user_codes,
            lambda_i
        )
        codes_user_nega1 = self.user_user_embedding(
            self.big_outfits_posi_feat,
            self.small_outfits_nega_feat,
            self.big_user_codes,
            lambda_i
        )
        codes_user_nega2 = self.user_user_embedding(
            self.big_outfits_nega_feat,
            self.small_outfits_nega_feat,
            self.big_user_codes,
            lambda_i
        )
        epsilon = 0.5
        codes_user = [
            (posi - epsilon * (nega1 + nega2)) / (1 + 2 * epsilon)
            for posi, nega1, nega2 in zip(codes_user_posi, codes_user_nega1, codes_user_nega2)]
        time_points.append(time.time())
        LOGGER.info(f'user_embedding() user-user embedding time cost: {time_points[-1] - time_points[-2]}s.')
        codes_item_posi = self.user_item_embedding(self.small_outfits_posi_feat, lambda_u)
        codes_item_nega = self.user_item_embedding(self.small_outfits_nega_feat, lambda_u)
        codes_item = [(posi - nega) / 2 for posi, nega in zip(codes_item_posi, codes_item_nega)]
        time_points.append(time.time())
        LOGGER.info(f'user_embedding() user-item embedding time cost: {time_points[-1] - time_points[-2]}s.')
        user_codes = [
            self.sign(self.beta * code_user + (1 - self.beta) * code_item).view(-1)
            for code_user, code_item in zip(codes_user, codes_item)
        ]
        time_points.append(time.time())
        LOGGER.info(f'user_embedding() total time cost: {time_points[-1] - time_points[0]}s.')
        return user_codes

    def forward(self, *inputs):
        """Forward.

        Return the scores for items.
        """
        items, uidx = inputs
        users = torch.stack([self.user_codes[u] for u in uidx])
        users = utils.to_device(users, self.device)
        # print(users.shape)
        # print(type(self.user_codes[0]))
        # print(self.user_codes[0].shape)
        # print(users)
        if self.param.use_semantic and self.param.use_visual:
            score_v = self.visual_output(users, items[0])
            score_s = self.semantic_output(users, items[1])
            score = [0.5 * (v + s) for v, s in zip(score_v, score_s)]
        elif self.param.use_visual:
            score = self.visual_output(users, items)
        elif self.param.use_semantic:
            score = self.semantic_output(users, items)
        else:
            raise ValueError
        return score