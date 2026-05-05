import argparse
import json
import os

import torch
import torchview

from config_networks import CustomNN

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_random_input(batch_size, input_shape, seed=torch.tensor(0)):
    """
    generates random input of a particular shape
    ensures gradients flow to seed
    """
    if type(input_shape[0]) is int:
        return torch.normal(0, 1, (batch_size,) + tuple(input_shape)) + seed
    else:
        return tuple(generate_random_input(batch_size, s, seed=seed) for s in input_shape)


class INPUT_TENSOR(torch.nn.Module):
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


p = argparse.ArgumentParser()
p.add_argument(
    "--config_json",
    nargs="+",
    type=str,
    default=[os.path.join(DIR, "net_configs", filename) for filename in os.listdir(os.path.join(DIR, "net_configs")) if filename.endswith(".json")],
    help="config files to display",
)
p.add_argument("--output_dir", type=str, default=os.path.join(DIR, "output"), help="directory to output files")
p.add_argument("--recursion_depth", type=int, default=1000, help="depth to expand nested torch modules")
args = p.parse_args()

for filename in args.config_json:
    model_name, _ = os.path.basename(filename).split(".")
    with open(filename) as f:
        structrue = json.load(f)
    model = CustomNN(structrue)
    x = generate_random_input(1, structrue["input_shape"])
    try:
        model_graph = torchview.draw_graph(
            model,
            input_data=x,
            expand_nested=True,
            save_graph=True,
            directory=args.output_dir,
            filename=f"visualize_{model_name}",
            depth=args.recursion_depth,
            device="cpu",
        )
    except RuntimeError:
        model = torch.nn.Sequential(INPUT_TENSOR(input_shape=structrue["input_shape"]), model)
        x = torch.rand(1)
        model_graph = torchview.draw_graph(
            model,
            input_data=x,
            expand_nested=True,
            save_graph=True,
            directory=args.output_dir,
            filename=f"visualize_{model_name}",
            depth=args.recursion_depth,
            device="cpu",
        )
