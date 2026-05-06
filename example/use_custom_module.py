import os

import torch
import torchview

from config_networks import CustomNN


class Invert(torch.nn.Module):
    def forward(self, X):
        return -X


input_shape = (6, 9, 4, 20)
structure = {
    "input_shape": input_shape,
    "layers": [
        {
            "type": "split",
            "combination": "sum",
            "branches": [
                None,
                [{"type": "custom", "module": Invert, "output_shape": input_shape}],
            ],
        }
    ],
}
model = CustomNN(structure)

print(model(torch.rand(input_shape)))  # should be all zeros, since this is x + (-x)

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_graph = torchview.draw_graph(
    model,
    input_data=torch.rand(input_shape),
    expand_nested=True,
    save_graph=True,
    directory=os.path.join(DIR, "images"),
    filename="visualize_custom",
    depth=1000,
    device="cpu",
)
