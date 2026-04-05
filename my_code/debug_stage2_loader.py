
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from zooloaders.ldmloader import ZooDataModule

dm = ZooDataModule(
    dataset="mnist",
    data_dir=r"D:/DataWerehouse/model_zoo/dnnwg_mnist",
    data_root=r"D:/DataWerehouse/model_zoo/dummy",
    batch_size=2,
    num_workers=0,
    num_sample=5,
    topk=None,
    normalize=False,
    scale=1.0,
)

dm.setup("fit")
batch = next(iter(dm.train_dataloader()))

print(batch.keys())
print(type(batch["weight"]), batch["weight"].shape)
print(type(batch["dataset"]), len(batch["dataset"]))
print(type(batch["dataset"][0]), batch["dataset"][0].shape)

print(batch["weight"].shape)
print(batch["dataset"].shape)