import torch
import torch.nn as nn
import onnx
import os
import json
import numpy as np

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def export_model(model, input_tensor, name):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    onnx_path = os.path.join(FIXTURE_DIR, f"{name}.onnx")
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=False,
    )

    meta = {
        "model": onnx_path,
        "input_shape": list(input_tensor.shape),
        "input_values": input_tensor.flatten().tolist(),
        "output_shape": list(output.shape),
        "output_values": output.flatten().tolist(),
    }

    meta_path = os.path.join(FIXTURE_DIR, f"{name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported {name}: {input_tensor.shape} -> {output.shape}")
    print(f"  Output: {output.flatten().tolist()}")
    return meta


# ── Model 1: Linear + ReLU ─────────────────────────────────
# Simple 2-layer MLP with ReLU activation
class LinearReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


torch.manual_seed(42)
model1 = LinearReLU()
# Set deterministic weights
with torch.no_grad():
    model1.fc1.weight.fill_(0.5)
    model1.fc1.bias.fill_(0.1)
    model1.fc2.weight.fill_(0.3)
    model2_fc2_bias = [0.2, -0.1]
    model1.fc2.bias = nn.Parameter(torch.tensor(model2_fc2_bias))

input1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
meta1 = export_model(model1, input1, "linear_relu")


# ── Model 2: Linear + ReLU + Softmax (classifier) ──────────
class LinearReLUSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 4, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


model2 = LinearReLUSoftmax()
with torch.no_grad():
    model2.fc1.weight.fill_(0.2)
    model2.fc1.bias.fill_(0.05)
    model2.fc2.weight.fill_(0.4)
    model2.fc2.bias.fill_(0.1)

input2 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
meta2 = export_model(model2, input2, "linear_relu_softmax")


# ── Model 3: Deeper MLP (3 Linear + 2 ReLU) ────────────────
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 6, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(6, 4, bias=True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4, 2, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


model3 = DeepMLP()
with torch.no_grad():
    model3.fc1.weight.fill_(0.1)
    model3.fc1.bias.fill_(0.0)
    model3.fc2.weight.fill_(0.2)
    model3.fc2.bias.fill_(0.05)
    model3.fc3.weight.fill_(0.3)
    model3.fc3.bias.fill_(-0.1)

input3 = torch.tensor([[0.5, 1.5, 2.5]])
meta3 = export_model(model3, input3, "deep_mlp")

print("\nAll models exported successfully!")
