from pathlib import Path
import torch

SRC = Path(r"D:\DataWerehouse\model_zoo\dataset_mnist_flat.pt")
DST = Path(r"D:\DataWerehouse\model_zoo\dnnwg_mnist_cnn.pt")

data = torch.load(SRC, map_location="cpu", weights_only=False)

train = data["trainset"].float()
val = data["valset"].float()
test = data["testset"].float()

print("train:", train.shape)
print("val  :", val.shape)
print("test :", test.shape)

assert train.ndim == 2 and train.shape[1] == 2464
assert val.ndim == 2 and val.shape[1] == 2464
assert test.ndim == 2 and test.shape[1] == 2464

# simplest payload: one dataset name mapped to tensors
payload = {
    "mnist_cnn": {
        "train": train,
        "val": val,
        "test": test,
        "weight_dim": train.shape[1],
    }
}

torch.save(payload, DST)
print("saved:", DST)