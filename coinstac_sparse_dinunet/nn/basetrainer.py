"""
@author: Aashis Khanal
@email: sraashis@gmail.com
"""

from collections import OrderedDict as _ODict
import os as _os
import torch as _torch

import coinstac_sparse_dinunet.config as _conf
import coinstac_sparse_dinunet.metrics as _base_metrics
import coinstac_sparse_dinunet.utils as _utils
import coinstac_sparse_dinunet.utils.tensorutils as _tu
import coinstac_sparse_dinunet.vision.plotter as _plot
from coinstac_sparse_dinunet.config.keys import *
from coinstac_sparse_dinunet.utils import stop_training_
from coinstac_sparse_dinunet.utils.logger import *
import torch
import torch as _torch
import copy
import torch.nn.functional as F
import torch.nn as nn
import types


class NNTrainer:
    def __init__(self, data_handle=None, **kw):
        self.cache = data_handle.cache
        self.input = _utils.FrozenDict(data_handle.input)
        self.state = _utils.FrozenDict(data_handle.state)
        self.nn = _ODict()
        self.device = _ODict()
        self.optimizer = _ODict()
        self.data_handle = data_handle

    def _init_nn_model(self):
        r"""
        User cam override and initialize required models in self.distrib dict.
        """
        raise NotImplementedError('Must be implemented in child class.')

    def _init_nn_weights(self, **kw):
        r"""
        By default, will initialize network with Kaimming initialization.
        If path to pretrained weights are given, it will be used instead.
        """

        if self.cache.get('pretrained_path') is not None:
            self.load_checkpoint(self.cache['pretrained_path'])
        elif self.cache['mode'] == Mode.TRAIN:
            _torch.manual_seed(self.cache['seed'])
            for mk in self.nn:
                _tu.initialize_weights(self.nn[mk])

    def _init_optimizer(self):
        r"""
        Initialize required optimizers here. Default is Adam,
        """
        first_model = list(self.nn.keys())[0]
        self.optimizer['adam'] = _torch.optim.Adam(self.nn[first_model].parameters(),
                                                   lr=self.cache['learning_rate'])

    def init_nn(self, init_model=False, init_optim=False, set_devices=False, init_weights=False):
        if init_model: self._init_nn_model()
        if init_optim: self._init_optimizer()
        if init_weights: self._init_nn_weights(init_weights=init_weights)
        if set_devices: self._set_gpus()

    def snip_forward_linear(self, params, x):

        return F.linear(x, params.weight * params.weight_mask, params.bias)

    def snip_forward_conv2d(self, params, x):

        return F.conv2d(x, params.weight * params.weight_mask, params.bias,
                        params.stride, params.padding, params.dilation, params.groups)

    def apply_mask_to_model(self, model, mask):

        prunable_layers = filter(
            lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
            model.modules()
        )
        for layer, keep_mask in zip(prunable_layers, mask):
            assert (layer.weight.shape == keep_mask.shape)
            # Set the masked weights to zero (NB the biases are ignored)
            layer.weight.data[keep_mask == 0.] = 0.

        return model

    @staticmethod
    def _forward_pre_hook(module, x):
        module.mask.requires_grad_(False)
        mask = module.mask
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        module.weight.data.mul_(mask.to(module.weight.to(device)))

    def register_pre_hook_mask(self, masks=None):
        masks_count = 0
        if masks is not None:
            print("Registering Mask!")
        assert masks is not None, 'Masks should be generated first.'
        for model_key in self.nn:
            for name, module in self.nn[model_key].named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    module.mask = nn.Parameter(masks[masks_count]).requires_grad_(False).to(
                        module.weight.to(self.device['gpu']))
                    masks_count += 1
                    module.register_forward_pre_hook(self._forward_pre_hook)

    # def apply_mask_to_model(self, model, mask):
    #     prunable_layers = filter(
    #         lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
    #         model.modules()
    #     )
    #     # print(torch.sum(torch.cat([torch.flatten(x == 1) for x in mask])))
    #     total_non_zero = 0
    #     for layer, keep_mask in zip(prunable_layers, mask):
    #         assert (layer.weight.shape == keep_mask.shape)
    #
    #         def hook_factory(keep_mask):
    #             """
    #             The hook function can't be defined directly here because of Python's
    #             late binding which would result in all hooks getting the very last
    #             mask! Getting it through another function forces early binding.
    #             """
    #
    #             def hook(grads):
    #                 return grads * keep_mask
    #
    #             return hook
    #
    #         # Step 1: Set the masked weights to zero (NB the biases are ignored)
    #         # Step 2: Make sure their gradients remain zero
    #         layer.weight.data[keep_mask == 0.] = 0.
    #         total_non_zero += torch.sum(torch.cat([torch.flatten(x != 0.) for x in layer.weight.data]))
    #         layer.weight.register_hook(hook_factory(keep_mask))
    #
    #     return model

    def apply_snip_pruning(self, dataset_cls):
        train_dataset = self.data_handle.get_train_dataset_for_masking(dataset_cls=dataset_cls)
        loader = self.data_handle.get_loader('train', dataset=train_dataset, drop_last=True, shuffle=True)
        mini_batch = next(iter(loader))  # inputs, labels, ix
        # inputs, labels = mini_batch['inputs'], mini_batch['labels']
        for model_key in self.nn:
            net = copy.deepcopy(self.nn[model_key])
            device = next(iter(self.nn[model_key].parameters())).device
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    layer.weight_mask = nn.Parameter(_torch.ones_like(layer.weight)).to(device)
                    nn.init.xavier_normal(layer.weight)
                    layer.weight.requires_grad = False

                if isinstance(layer, nn.Linear):
                    layer.forward = types.MethodType(self.snip_forward_linear, layer)

                if isinstance(layer, nn.Conv2d):
                    layer.forward = types.MethodType(self.snip_forward_conv2d, layer)

            net.to(device)
            net.zero_grad()
            it = self.single_iteration_for_masking(net, mini_batch)
            sparsity_level = abs(1 - it['sparsity_level'])
            it['loss'].backward()

            grads_abs = []

            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    grads_abs.append(torch.abs(layer.weight_mask.grad))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
            norm_factor = torch.sum(all_scores)
            all_scores.div_(norm_factor)

            num_params_to_keep = int(len(all_scores) * sparsity_level)
            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]
            keep_masks = []
            for g in grads_abs:
                keep_masks.append(((g / norm_factor) >= acceptable_score).float())

            # initialize_mult = []
            # for i in range(len(grads_abs)):
            #     initialize_mult.append(grads_abs[i] / norm_factor)

            self.nn[model_key] = self.apply_mask_to_model(self.nn[model_key], keep_masks)
            return keep_masks

    def _set_gpus(self):
        self.device['gpu'] = _torch.device("cpu")
        if self.cache.get('gpus') is not None and len(self.cache['gpus']) > 0:
            if _conf.CUDA_AVAILABLE:
                self.device['gpu'] = _torch.device(f"cuda:{self.cache['gpus'][0]}")
                if len(self.cache['gpus']) >= 2:
                    for mkey in self.nn:
                        self.nn[mkey] = _torch.nn.DataParallel(self.nn[mkey], self.cache['gpus'])
            else:
                pass
                # raise Exception(f'*** GPU not detected in {self.state["clientId"]}. ***')
        for model_key in self.nn:
            self.nn[model_key] = self.nn[model_key].to(self.device['gpu'])

    def load_checkpoint(self, file_path):
        try:
            chk = _torch.load(file_path)
        except:
            chk = _torch.load(file_path, map_location='cpu')

        if chk.get('source', 'Unknown').lower() == 'coinstac':
            for m in chk['models']:
                try:
                    self.nn[m].module.load_state_dict(chk['models'][m])
                except:
                    self.nn[m].load_state_dict(chk['models'][m])

            for m in chk['optimizers']:
                try:
                    self.optimizer[m].module.load_state_dict(chk['optimizers'][m])
                except:
                    self.optimizer[m].load_state_dict(chk['optimizers'][m])
        else:
            mkey = list(self.nn.keys())[0]
            try:
                self.nn[mkey].module.load_state_dict(chk)
            except:
                self.nn[mkey].load_state_dict(chk)

    def save_checkpoint(self, file_path, src='coinstac'):
        checkpoint = {'source': src}
        for k in self.nn:
            checkpoint['models'] = {}
            try:
                checkpoint['models'][k] = self.nn[k].module.state_dict()
            except:
                checkpoint['models'][k] = self.nn[k].state_dict()
        for k in self.optimizer:
            checkpoint['optimizers'] = {}
            try:
                checkpoint['optimizers'][k] = self.optimizer[k].module.state_dict()
            except:
                checkpoint['optimizers'][k] = self.optimizer[k].state_dict()
        _torch.save(checkpoint, file_path)

    def evaluation(self, mode='eval', dataset_list=None, save_pred=False, use_padded_sampler=False):
        for k in self.nn:
            self.nn[k].eval()

        eval_avg, eval_metrics = self.new_averages(), self.new_metrics()
        eval_loaders = []

        for d in dataset_list:
            if d and len(d) > 0:
                eval_loaders.append(
                    self.data_handle.get_loader(handle_key=mode, dataset=d, shuffle=False,
                                                use_padded_sampler=use_padded_sampler)
                )

        def _update_scores(_out, _it, _avg, _metrics):
            if _out is None:
                _out = {}
            _avg.accumulate(_out.get('averages', _it['averages']))
            _metrics.accumulate(_out.get('metrics', _it['metrics']))

        with _torch.no_grad():
            for loader in eval_loaders:
                its = []
                metrics = self.new_metrics()
                avg = self.new_averages()

                for i, batch in enumerate(loader, 1):
                    it = self.iteration(batch)

                    if save_pred:
                        if self.cache['load_sparse']:
                            its.append(it)
                        else:
                            _update_scores(self.save_predictions(loader.dataset, it), it, avg, metrics)
                    else:
                        _update_scores(None, it, avg, metrics)

                    if self.cache['verbose'] and len(eval_loaders) <= 1 and lazy_debug(i):
                        info(
                            f" Itr:{i}/{len(loader)}, "
                            f"Averages:{it.get('averages').get()}, Metrics:{it.get('metrics').get()}"
                        )

                if save_pred and self.cache['load_sparse']:
                    its = self.reduce_iteration(its)
                    _update_scores(self.save_predictions(loader.dataset, its), its, avg, metrics)

                if self.cache['verbose'] and len(eval_loaders) > 1:
                    info(f" {mode}, {avg.get()}, {metrics.get()}")

                eval_metrics.accumulate(metrics)
                eval_avg.accumulate(avg)

            info(f"{mode} metrics: {eval_avg.get()}, {eval_metrics.get()}", self.cache.get('verbose'))
            return eval_avg, eval_metrics

    def training_iteration_local(self, i, batch):
        r"""
        Learning step for one batch.
        We decoupled it so that user could implement any complex/multi/alternate training strategies.
        """
        it = self.iteration(batch)
        it['loss'].backward()
        if i % self.cache.get('local_iterations', 1) == 0:
            first_optim = list(self.optimizer.keys())[0]
            self.optimizer[first_optim].step()
            self.optimizer[first_optim].zero_grad()
        return it

    def init_training_cache(self):
        self.cache[Key.TRAIN_LOG] = []
        self.cache[Key.VALIDATION_LOG] = []
        self.cache['best_val_epoch'] = 0
        self.cache.update(best_val_score=0.0 if self.cache['metric_direction'] == 'maximize' else _conf.max_size)

    def train_local(self, train_dataset, val_dataset):
        out = {}

        if not isinstance(val_dataset, list):
            val_dataset = [val_dataset]

        loader = self.data_handle.get_loader('train', dataset=train_dataset, drop_last=True, shuffle=True)
        local_iter = self.cache.get('local_iterations', 1)
        tot_iter = len(loader) // local_iter
        for ep in range(1, self.cache['epochs'] + 1):
            for k in self.nn:
                self.nn[k].train()

            _metrics, _avg = self.new_metrics(), self.new_averages()
            ep_avg, ep_metrics, its = self.new_averages(), self.new_metrics(), []

            for i, batch in enumerate(loader, 1):
                its.append(self.training_iteration_local(i, batch))
                if i % local_iter == 0:
                    it = self.reduce_iteration(its)

                    ep_avg.accumulate(it['averages']), ep_metrics.accumulate(it['metrics'])
                    _avg.accumulate(it['averages']), _metrics.accumulate(it['metrics'])

                    _i, its = i // local_iter, []
                    if lazy_debug(_i) or _i == tot_iter:
                        info(f"Ep:{ep}/{self.cache['epochs']},Itr:{_i}/{tot_iter},{_avg.get()},{_metrics.get()}",
                             self.cache.get('verbose'))
                        self.cache[Key.TRAIN_LOG].append([*_avg.get(), *_metrics.get()])
                        _metrics.reset(), _avg.reset()
                    self.on_iteration_end(i=_i, ep=ep, it=it)

            if val_dataset and ep % self.cache.get('validation_epochs', 1) == 0:
                info('--- Validation ---', self.cache.get('verbose'))
                val_averages, val_metric = self.evaluation(mode='validation', dataset_list=val_dataset,
                                                           use_padded_sampler=True)
                self.cache[Key.VALIDATION_LOG].append([*val_averages.get(), *val_metric.get()])
                out.update(**self._save_if_better(ep, val_metric))

                self._on_epoch_end(ep=ep, ep_averages=ep_avg, ep_metrics=ep_metrics,
                                   val_averages=val_averages, val_metrics=val_metric)

                if lazy_debug(ep):
                    self._save_progress(self.cache, epoch=ep)

                if self._stop_early(ep, val_metric, val_averages=val_averages,
                                    epoch_averages=ep_avg, epoch_metrics=ep_metrics):
                    break

        self._save_progress(self.cache, epoch=ep)
        _utils.save_cache(self.cache, self.cache['log_dir'])
        return out

    def iteration(self, batch):
        r"""
        Left for user to implement one mini-bath iteration:
        Example:{
                    inputs = batch['input'].to(self.device['gpu']).float()
                    labels = batch['label'].to(self.device['gpu']).long()
                    out = self.distrib['model'](inputs)
                    loss = F.cross_entropy(out, labels)
                    out = F.softmax(out, 1)
                    _, pred = torch.max(out, 1)
                    sc = self.new_metrics()
                    sc.add(pred, labels)
                    avg = self.new_averages()
                    avg.add(loss.item(), len(inputs))
                    return {'loss': loss, 'averages': avg, 'output': out, 'metrics': sc, 'predictions': pred}
                }
        Note: loss, averages, and metrics are required, whereas others are optional
            -we will have to do backward on loss
            -we need to keep track of loss
            -we need to keep track of metrics
        """
        return {}

    def single_iteration_for_masking(self, model, batch):
        r"""
        Left for user to implement one mini-bath iteration:
        Example:{
                    inputs = batch['input'].to(self.device['gpu']).float()
                    labels = batch['label'].to(self.device['gpu']).long()
                    out = self.distrib['model'](inputs)
                    loss = F.cross_entropy(out, labels)
                    out = F.softmax(out, 1)
                    _, pred = torch.max(out, 1)
                    sc = self.new_metrics()
                    sc.add(pred, labels)
                    avg = self.new_averages()
                    avg.add(loss.item(), len(inputs))
                    return {'loss': loss, 'averages': avg, 'output': out, 'metrics': sc, 'predictions': pred}
                }
        Note: loss, averages, and metrics are required, whereas others are optional
            -we will have to do backward on loss
            -we need to keep track of loss
            -we need to keep track of metrics
        """
        return {}

    def save_predictions(self, dataset, its):
        pass

    def reduce_iteration(self, its):
        reduced = {}.fromkeys(its[0].keys(), None)
        for key in reduced:
            if isinstance(its[0][key], _base_metrics.COINNAverages):
                reduced[key] = self.new_averages()
                [reduced[key].accumulate(ik[key]) for ik in its]

            elif isinstance(its[0][key], _base_metrics.COINNMetrics):
                reduced[key] = self.new_metrics()
                [reduced[key].accumulate(ik[key]) for ik in its]
            else:
                def collect(k=key, src=its):
                    _data = []
                    is_tensor = isinstance(src[0][k], _torch.Tensor)
                    is_tensor = is_tensor and not src[0][k].requires_grad and src[0][k].is_leaf
                    for ik in src:
                        if is_tensor:
                            _data.append(ik[k] if len(ik[k].shape) > 0 else ik[k].unsqueeze(0))
                        else:
                            _data.append(ik[k])
                    if is_tensor:
                        return _torch.cat(_data)
                    return _data

                reduced[key] = collect

        return reduced

    def _save_if_better(self, epoch, val_metrics):
        return {}

    def new_metrics(self):
        return _base_metrics.COINNMetrics()

    def new_averages(self):
        return _base_metrics.COINNAverages(num_averages=1)

    def _on_epoch_end(self, ep, **kw):
        r"""
        Any logic to run after an epoch ends.
        """
        return {}

    def on_iteration_end(self, i, ep, it):
        r"""
        Any logic to run after an iteration ends.
        """
        return {}

    def _save_progress(self, cache, epoch):
        _plot.plot_progress(cache, self.cache['log_dir'], plot_keys=[Key.TRAIN_LOG], epoch=epoch)
        _plot.plot_progress(cache, self.cache['log_dir'], plot_keys=[Key.VALIDATION_LOG],
                            epoch=epoch // self.cache['validation_epochs'])

    def _stop_early(self, epoch, val_metrics=None, **kw):
        return stop_training_(epoch, self.cache)
