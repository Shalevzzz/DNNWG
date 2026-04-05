import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== YOUR CNN ====
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.ModuleList([
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 4, 2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(36, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        ])

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x


# ==== PARAM MAPPING ====
PARAMS = [
    ("module_list.0.weight", (8,1,5,5)),
    ("module_list.0.bias",   (8,)),
    ("module_list.3.weight", (6,8,5,5)),
    ("module_list.3.bias",   (6,)),
    ("module_list.6.weight", (4,6,2,2)),
    ("module_list.6.bias",   (4,)),
    ("module_list.9.weight", (20,36)),
    ("module_list.9.bias",   (20,)),
    ("module_list.11.weight",(10,20)),
    ("module_list.11.bias",  (10,))
]

def vec_to_state(vec):
    vec = vec[:2464]  # remove padding
    state = {}
    i = 0
    for name, shape in PARAMS:
        n = torch.tensor(shape).prod().item()
        state[name] = vec[i:i+n].view(shape)
        i += n
    return state


# ==== DATA ====
test_loader = DataLoader(
    datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor()),
    batch_size=256
)

def eval_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


# ==== LOAD SAMPLES ====
samples = torch.load("generated_weights_mnist.pt")

accs = []
for i in range(samples.shape[0]):
    net = Net().to(DEVICE)
    state = vec_to_state(samples[i])
    net.load_state_dict(state)

    acc = eval_model(net)
    accs.append(acc)

    print(f"sample {i}: acc = {acc:.4f}")

print("mean:", sum(accs)/len(accs))
print("max:", max(accs))