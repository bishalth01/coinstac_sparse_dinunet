"""
@author: Bishal Thapaliya
@email: bthapaliya16@gmail.com
"""

import datetime as _dt
import time as _t

from coinstac_sparse_dinunet import config as _conf
import torch


def performance_improved_(epoch, score, cache):
    delta = cache.get('score_delta', _conf.score_delta)
    improved = False
    if cache['metric_direction'] == 'maximize':
        improved = score > cache['best_val_score'] + delta
    elif cache['metric_direction'] == 'minimize':
        improved = score < cache['best_val_score'] - delta

    if improved:
        cache['best_val_epoch'] = epoch
        cache['best_val_score'] = score
    return bool(improved)


def stop_training_(epoch, cache):
    return epoch - cache['best_val_epoch'] > cache.get('patience', cache['epochs'])

def get_model_sps(model):
    nonzero = total = 0
    for name, param in model.named_parameters():
        if 'mask' not in name:
            nz_count = torch.count_nonzero(param).item()
            total_params = param.numel()
            nonzero += nz_count
            total += total_params
    abs_sps = 100 * (total - nonzero) / total
    return abs_sps


def duration(cache: dict, begin, key):
    t_del = _dt.datetime.fromtimestamp(_t.time()) - _dt.datetime.fromtimestamp(begin)
    if cache.get(key) is None:
        cache[key] = [t_del.total_seconds()]
    else:
        cache[key].append(t_del.total_seconds())
    return t_del
