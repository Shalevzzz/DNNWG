import torch
from pathlib import Path

p = Path(r"D:\DataWerehouse\model_zoo\dnnwg_mnist_cnn.pt")
data = torch.load(p, map_location="cpu", weights_only=False)

print(type(data))
print(data.keys())

mnist = data["mnist_cnn"]
print(mnist.keys())
print("train:", mnist["train"].shape)
print("val  :", mnist["val"].shape)
print("test :", mnist["test"].shape)
print("weight_dim:", mnist["weight_dim"])