import torch
import random
import numpy as np
from omegaconf import OmegaConf


def move_to_device(dct, device):
    for key, value in dct.items():
        if isinstance(value, torch.Tensor):
            dct[key] = value.to(device)
    return dct

def slice_trajdict_with_t(data_dict, start_idx=0, end_idx=None, step=1):
    if end_idx is None:
        end_idx = max(arr.shape[1] for arr in data_dict.values())
    return {key: arr[:, start_idx:end_idx:step, ...] for key, arr in data_dict.items()}

def aggregate_dct(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct

def cfg_to_dict(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    for key in cfg_dict:
        if isinstance(cfg_dict[key], list):
            cfg_dict[key] = ",".join(cfg_dict[key])
    return cfg_dict

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
