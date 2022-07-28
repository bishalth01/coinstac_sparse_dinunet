"""
@author: Bishal Thapaliya
@email: bthapaliya16@gmail.com
"""

from os import sep as _sep
import torch as _torch

from coinstac_sparse_dinunet import config as _conf
from coinstac_sparse_dinunet.utils import tensorutils as _tu
import numpy as _np
import coinstac_sparse_dinunet.utils as _utils


class COINNLearner:
    def __init__(self, trainer=None, mp_pool=None, **kw):
        self.cache = trainer.cache
        self.input = trainer.input
        self.state = trainer.state
        self.trainer = trainer
        self.global_modes = self.input.get('global_modes', {})
        self.pool = mp_pool
        self.dtype = f"float{self.cache['precision_bits']}"
        self.device = trainer.device["gpu"]

    def step(self) -> dict:
        out = {}
        grads = _tu.load_arrays(self.state['baseDirectory'] + _sep + self.input['avg_grads_file'])
        grad_index = 0
        first_model = list(self.trainer.nn.keys())[0]

        for i, param in enumerate(self.trainer.nn[first_model].named_parameters()):
            if 'mask' not in param[0]:
                param[1].grad = _torch.tensor(grads[grad_index], dtype=_torch.float32).to(self.device)
                grad_index += 1

        first_optim = list(self.trainer.optimizer.keys())[0]
        self.trainer.optimizer[first_optim].step()
        return out

    def backward(self):
        out = {}
        first_model = list(self.trainer.nn.keys())[0]
        first_optim = list(self.trainer.optimizer.keys())[0]

        self.trainer.nn[first_model].train()
        self.trainer.optimizer[first_optim].zero_grad()
        its = []
        for _ in range(self.cache['local_iterations']):
            batch, nxt_iter_out = self.trainer.data_handle.next_iter()
            it = self.trainer.iteration(batch)
            it['loss'].backward()
            its.append(it)
            out.update(**nxt_iter_out)
        return self.trainer.reduce_iteration(its), out

    def to_reduce(self):
        it, out = self.backward()
        first_model = list(self.trainer.nn.keys())[0]
        out['grads_file'] = _conf.grads_file
        grads = _tu.extract_grads(self.trainer.nn[first_model], dtype=self.dtype)
        _tu.save_arrays(
            self.state['transferDirectory'] + _sep + out['grads_file'],
            _np.array(grads, dtype=object)
        )
        out['reduce'] = True
        return it, out
