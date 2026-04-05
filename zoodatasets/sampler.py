import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def matpadder(x, max_in=512):
    delta2 = max_in - x.shape[1]
    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out


class ZooDataset(Dataset):
    """weights dataset for stage 2."""

    def __init__(
        self,
        root='zoodata',
        dataset='joint',
        split='train',
        scale=1.0,
        num_sample=5,
        topk=None,
        transform=None,
        normalize=False,
        max_len=2864,
        cond_path='clip_encode_dsets_20_cond_.pt',
    ):
        super().__init__()
        self.dataset = dataset
        self.topk = topk
        self.max_len = max_len
        self.split = split
        self.normalize = normalize
        self.num_sample = num_sample
        self.scale = scale
        self.root = root
        self.transform = transform

        datapath = os.path.join(root, f'weights/{split}_data')
        self.data = self.load_data(datapath)

        # load all condition data once
        self.cond_data = torch.load(cond_path, map_location="cpu", weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        weights = self.data[idx]
        keys = list(weights)

        conds = []
        weight = None

        for k in keys:
            w = weights[k] / self.scale   # [1, D]
            if len(w.shape) < 2:
                w = w.unsqueeze(0)
            if w.shape[1] < self.max_len:
                w = matpadder(w, self.max_len)

            x = self.cond_data[k]   # e.g. self.cond_data["mnist"]

            classes = list(range(len(x)))
            cdata = []
            for cls in classes:
                cx = x[cls]                 # [20, 512]
                ridx = torch.randperm(len(cx))
                cdata.append(cx[ridx][:self.num_sample])

            conds.append(torch.stack(cdata, 0).type(torch.float32))  # [10, 5, 512]
            weight = w

        sample = {
            "weight": weight,      # [1, 2864]
            "dataset": conds       # [ [10, 5, 512] ]
        }
        return sample

    def load_data(self, file):
        data = torch.load(file, map_location="cpu", weights_only=False)
        xdata = []

        keys = list(data)
        for k in keys:
            w = data[k][0].detach().cpu()

            if len(w.shape) < 2:
                w = w.unsqueeze(0)

            # split into one sample per row/model
            for i in range(w.shape[0]):
                xdata.append({k: w[i:i+1]})   # keep shape [1, D]

        return xdata