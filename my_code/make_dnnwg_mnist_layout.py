from pathlib import Path
import torch

SRC = Path(r"D:\DataWerehouse\model_zoo\dataset_mnist_flat.pt")
DST_ROOT = Path(r"D:\DataWerehouse\model_zoo\dnnwg_mnist")
WEIGHTS_DIR = DST_ROOT / "weights"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

data = torch.load(SRC, map_location="cpu", weights_only=False)

train = data["trainset"].float()
val = data["valset"].float()
test = data["testset"].float()

print("train:", train.shape)
print("val  :", val.shape)
print("test :", test.shape)

# DNNWG's ZooDataset expects: data[k][0] -> tensor [N, D]
# So each file should look like:
# {"mnist": [tensor]}
torch.save({"mnist": [train]}, WEIGHTS_DIR / "train_data.pt")
torch.save({"mnist": [val]},   WEIGHTS_DIR / "val_data.pt")
torch.save({"mnist": [test]},  WEIGHTS_DIR / "test_data.pt")

print("saved to:", WEIGHTS_DIR)