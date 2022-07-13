## coinstac-sparse-dinunet
#### Distributed Sparse Neural Network implementation  on COINSTAC.

![PyPi version](https://img.shields.io/pypi/v/coinstac-sparse-dinunet)
![versions](https://img.shields.io/pypi/pyversions/pybadges.svg)

```
pip install coinstac-sparse-dinunet
```
#### Specify supported packages like pytorch & torchvision in a requirements.txt file
#### Highlights:
```
1. Creates sparse network based on single shot pruning SNIP algorithm (https://arxiv.org/abs/1810.02340). 
2. Handles multi-network/complex training schemes. 
3. Automatic data splitting/k-fold cross validation.
4. Automatic model checkpointing.
5. GPU enabled local sites.
6. Customizable metrics(w/Auto serialization between nodes) to work with any schemes.
7. We can integrate any custom reduction and learning mechanism by extending coinstac_sparse_dinunet.distrib.reducer/learner.
8. Realtime profiling each sites by specifying in compspec file(see dinune_fsv example below for details). 
...
```


<hr />

[//]: # (![DINUNET]&#40;assets/dinunet.png&#41;)


[//]: # (### Working examples:)

[//]: # (1. **[FreeSurfer volumes classification.]&#40;https://github.com/trendscenter/dinunet_implementations/&#41;**)

[//]: # (2. **[VBM 3D images classification.]&#40;https://github.com/trendscenter/dinunet_implementations_gpu&#41;**)

### [Running an analysis](https://github.com/trendscenter/coinstac-instructions/blob/master/coinstac-how-to-run-analysis.md) in the coinstac App.
### Add a new NN computation to COINSTAC (Development guide):
#### imports

```python
from coinstac_sparse_dinunet import COINNDataset, COINNTrainer, COINNLocal
from coinstac_sparse_dinunet.metrics import COINNAverages, Prf1a
```

#### 1. Define Data Loader
```python
class MyDataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = None

    def load_index(self, id, file):
        data_dir = self.path(id, 'data_dir') # data_dir comes from inputspecs.json
        ...
        self.indices.append([id, file])

    def __getitem__(self, ix):
        id, file = self.indices[ix]
        data_dir = self.path(id, 'data_dir') # data_dir comes from inputspecs.json
        label_dir = self.path(id, 'label_dir') # label_dir comes from inputspecs.json
        ...
        # Logic to load, transform single data item.
        ...
        return {'inputs':.., 'labels': ...}
```

#### 2. Define Trainer
```python
class MyTrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['model'] = MYModel(in_size=self.cache['input_size'], out_size=self.cache['num_class'])
    
    
    def single_iteration_for_masking(self, model, batch):
        # Interation for masking. Defines specific output and loss functions for masking using SNIP.
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()
        indices = batch['ix'].to(self.device['gpu']).long()

        out = F.log_softmax(model(inputs), 1)
        loss = F.nll_loss(out, labels)

        return {'out': out, 'loss': loss, 'indices': indices}


    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()

        out = F.log_softmax(self.nn['model'](inputs), 1)
        loss = F.nll_loss(out, labels)
        _, predicted = torch.max(out, 1)
        score = self.new_metrics()
        score.add(predicted, labels)
        val = self.new_averages()
        val.add(loss.item(), len(inputs))
        return {'out': out, 'loss': loss, 'averages': val,
                'metrics': score, 'prediction': predicted}
```

<hr />

#### Advanced use cases:

* **Define custom metrics:**
  - Extend [coinstac_sparse_dinunet.metrics.COINNMetrics](https://github.com/bishalth01/coinstac_sparse_dinunet/blob/master/coinstac_sparse_dinunet/metrics/metrics.py)
  - Example: [coinstac_sparse_dinunet.metrics.Prf1a](https://github.com/bishalth01/coinstac_sparse_dinunet/blob/master/coinstac_sparse_dinunet/metrics/metrics.py) for Precision, Recall, F1, and Accuracy
  
* **Define [Custom Learner](https://github.com/bishalth01/coinstac_sparse_dinunet/blob/master/coinstac_sparse_dinunet/distrib/learner.py) / [custom Aggregator]/ [custom Aggregator](https://github.com/bishalth01/coinstac_sparse_dinunet/blob/master/coinstac_sparse_dinunet/distrib/reducer.py)  (Default is Distributed SGD)**



#### Referenced from Trends Center coinstac-dinunet repository (https://github.com/trendscenter/coinstac-dinunet)