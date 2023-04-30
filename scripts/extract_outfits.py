#!/usr/bin/env python
"""Script for training, evaluate and retrieval."""
import argparse
import logging
import os
import pickle
import shutil
import textwrap

import numpy as np
import torch
import tqdm
import yaml
from torch.nn.parallel import data_parallel

import polyvore
import utils

NAMED_SOLVERS = utils.get_named_class(polyvore.solver)
NAMED_MODELS = utils.get_named_class(polyvore.model)


def get_net(config):
    """Get network."""
    # get net param
    net_param = config.net_param
    LOGGER.info("Initializing %s", config.net_param.name)
    LOGGER.info(net_param)
    # dimension of latent codes
    net = NAMED_MODELS[config.net_param.name](net_param)
    # Load model from pre-trained file
    if config.load_trained:
        # load weights from pre-trained model
        num_devices = torch.cuda.device_count()
        map_location = {"cuda:{}".format(i): "cpu" for i in range(num_devices)}
        LOGGER.info("Loading pre-trained model from %s.", config.load_trained)
        state_dict = torch.load(config.load_trained, map_location=map_location)
        # when new user problem from pre-trained model
        if config.cold_start:
            # TODO: fit with new arch
            # reset the user's embedding
            LOGGER.info("Reset the user embedding")
            # TODO: use more decent way to load pre-trained model for new user
            weight = "user_embedding.encoder.weight"
            state_dict[weight] = torch.zeros(net_param.dim, net_param.num_users)
            net.load_state_dict(state_dict)
            net.user_embedding.init_weights()
        else:
            # load pre-trained model
            net.load_state_dict(state_dict)
    elif config.resume:  # resume training
        LOGGER.info("Training resume from %s.", config.resume)
    else:
        LOGGER.info("Loading weights from backbone %s.", net_param.backbone)
        net.init_weights()
    LOGGER.info("Copying net to GPU-%d", config.gpus[0])
    net.cuda(device=config.gpus[0])
    return net


def update_npz(fn, results):
    if fn is None:
        return
    if os.path.exists(fn):
        pre_results = dict(np.load(fn, allow_pickle=True))
        pre_results.update(results)
        results = pre_results
    np.savez(fn, **results)


def extract_features(config):
    LOGGER.info("Extract features.")
    data_param = config.data_param
    LOGGER.info("Dataset for positive tuples: %s", data_param)
    loader = polyvore.data.get_dataloader(data_param)
    net = get_net(config).eval()
    device = config.gpus[0]
    pbar = tqdm.tqdm(loader, desc="Computing features")
    user_codes = net.get_user_binary_code(device)
    item_codes = dict()
    outfits = []
    lambda_i, lambda_u, alpha = net.get_matching_weight()
    for inputv in pbar:
        items, tpls = inputv
        # _, text = items
        items = utils.to_device(items, device)
        with torch.no_grad():
            features = net.compute_codes(items)
        if data_param.use_semantic and data_param.use_visual:
            feat_v, feat_t = features
            feat_v = [feat.cpu().numpy().astype(np.int8) for feat in feat_v]
            feat_t = [feat.cpu().numpy().astype(np.int8) for feat in feat_t]
        elif data_param.use_semantic:
            feat_t = features
            feat_v = [[] for _ in feat_t]
            feat_t = [feat.cpu().numpy().astype(np.int8) for feat in feat_t]
        elif data_param.use_visual:
            feat_v = features
            feat_v = [feat.cpu().numpy().astype(np.int8) for feat in feat_v]
            feat_t = [[] for _ in feat_v]
        else:
            raise ValueError
        '''
        for n, tpl in enumerate(tpls):
            names = loader.dataset.get_names(tpl)
            for c, name in enumerate(names):
                item_codes[name] = [feat_v[c][n], feat_t[c][n]]
        '''
        for n, tpl in enumerate(tpls):
            outfit_v = sum([v[n] for v in feat_v]) / len(feat_v)
            outfit_t = sum([t[n] for t in feat_t]) / len(feat_t)
            outfit = (outfit_v + outfit_t) / 2
            names = loader.dataset.get_names(tpl)
            # outfits.append([names, text[n], outfit])
            outfits.append([names, outfit])

    print(len(outfits))
    with open(config.feature_file, "wb") as f:
        data = dict(
            user_codes=user_codes,
            item_codes=item_codes,
            lambda_u=lambda_u,
            lambda_i=lambda_i,
            alpha=alpha,
        )
        pickle.dump(data, f)


ACTION_FUNS = {
    "extract-features": extract_features,
}

LOGGER = logging.getLogger("polyvore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Fashion Hash Net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Fashion Hash Net Training Script
            --------------------------------
            Actions:
                1. train: train fashion net.
                2. evaluate: evaluate NDCG and accuracy.
                3. retrieval: retrieval for items.
                """
        ),
    )
    actions = ACTION_FUNS.keys()
    parser.add_argument("action", help="|".join(sorted(actions)))
    parser.add_argument("--cfg", help="configuration file.")
    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        kwargs = yaml.load(f, Loader=yaml.FullLoader)
    config = polyvore.param.FashionParam(**kwargs)
    # config.add_timestamp()
    logfile = utils.config_log(stream_level=config.log_level, log_file=config.log_file)
    LOGGER.info("Logging to file %s", logfile)
    LOGGER.info("Fashion param : %s", config)

    if args.action in actions:
        ACTION_FUNS[args.action](config)
        exit(0)
    else:
        LOGGER.info("Action %s is not in %s", args.action, "|".join(actions))
        exit(1)
