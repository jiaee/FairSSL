import random
import torch
from torch.utils.data import TensorDataset
from collections import defaultdict
import math

def load_split(pt_path):
    data = torch.load(pt_path)
    return TensorDataset(
        data['samples'],
        data['labels'], 
        data['gender'], 
        data['subject_ids']
    )

def drop_unnecessary(train_ds, val_ds, test_ds):
    X, y, g, sid = train_ds.tensors
    N, C, T = X.shape
    nan_map = torch.isnan(X).sum(0)
    full_nan_ch = (nan_map == N).all(1)
    keep_idx = (~full_nan_ch).nonzero(as_tuple=True)[0]
    print("Dropping channels:", full_nan_ch.nonzero(as_tuple=True)[0].tolist())
    def apply(ds):
        X_, y_, g_, sid_ = ds.tensors
        return TensorDataset(X_[:, keep_idx], y_, g_, sid_)
    return apply(train_ds), apply(val_ds), apply(test_ds)

def median_impute(ds, medians=None):
    X, y, g, sid = ds.tensors
    N, C, T = X.shape
    if medians is None:
        flat = X.permute(1, 0, 2).reshape(C, -1)
        medians = torch.nanmedian(flat, 1).values
    med_exp = medians.view(1, C, 1).expand_as(X)
    Xf = torch.where(torch.isnan(X), med_exp, X)
    return TensorDataset(Xf, y, g, sid), medians


def _group_indices_by_label(ds):
    """Return {label: [indices]} for a TensorDataset whose 2nd tensor is y."""
    label_to_idx = defaultdict(list)
    labels = ds.tensors[1] 
    for i, lab in enumerate(labels):
        label_to_idx[int(lab)].append(i)
    return label_to_idx

class SameLabelBatchSampler(torch.utils.data.Sampler):
    """
    Iterates through the dataset label-by-label, yielding batches that are
    homogeneous w.r.t. y.  
    """
    def __init__(self, dataset, batch_size):
        self.bs  = batch_size
        self.map = _group_indices_by_label(dataset)

    def __iter__(self):
        for lbl, idxs in self.map.items():
            random.shuffle(idxs)
            for i in range(0, len(idxs), self.bs):
                yield idxs[i:i + self.bs]

    def __len__(self):
        return sum(
            math.ceil(len(idxs) / self.bs) for idxs in self.map.values()
        )