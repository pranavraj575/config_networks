import json
import os

import torch
import torchview

from config_networks import CustomNN

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_random_input(batch_size, input_shape, seed=torch.tensor(0)):
    if type(input_shape[0]) is int:
        return torch.normal(0, 1, (batch_size,) + tuple(input_shape)) + seed
    else:
        return tuple(generate_random_input(batch_size, s, seed=seed) for s in input_shape)


class GenerateInput(torch.nn.Module):
    def __init__(self, input_shape, batch_size=1):
        super().__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size

    def forward(self, X):
        seed = torch.sum(X)
        return generate_random_input(
            batch_size=self.batch_size,
            input_shape=self.input_shape,
            seed=seed,
        )


recursion_depth = 100
directory = os.path.join(DIR, "output")
for filename in os.listdir(os.path.join(DIR, "net_configs")):
    if not filename.endswith(".json"):
        continue
    filename = os.path.join(DIR, "net_configs", filename)
    model_name, _ = os.path.basename(filename).split(".")
    with open(filename) as f:
        structrue = json.load(f)
    model = CustomNN(structrue)
    x = generate_random_input(1, structrue["input_shape"])
    try:
        model_graph = torchview.draw_graph(
            model, input_data=x, expand_nested=True, save_graph=True, directory=directory, filename=f"visualize_{model_name}", depth=recursion_depth
        )
    except RuntimeError:
        model = torch.nn.Sequential(GenerateInput(input_shape=structrue["input_shape"]), model)
        x = torch.rand(1)
        model_graph = torchview.draw_graph(
            model, input_data=x, expand_nested=True, save_graph=True, directory=directory, filename=f"visualize_{model_name}", depth=recursion_depth
        )
