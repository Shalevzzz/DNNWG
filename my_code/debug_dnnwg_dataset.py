import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from zooloaders.autoloader import ZooDataModule

dm = ZooDataModule(
    dataset="mnist",
    data_dir=r"D:/DataWerehouse/model_zoo/dnnwg_mnist",
    data_root=r"D:/DataWerehouse/model_zoo/dummy",
    batch_size=4,
    num_workers=0,
    scale=1.0,
    num_sample=5,
    topk=None,
    normalize=False,
)

dm.setup("fit")
batch = next(iter(dm.train_dataloader()))

print(type(batch))
print(batch.keys())
print(batch["weight"].shape)
print(batch["weight"].dtype)