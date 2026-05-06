import os

import torch
import torchview

from config_networks import CustomNN


class Scale(torch.nn.Module):
    def __init__(self, scalar=1):
        super().__init__()
        self.scalar = scalar

    def forward(self, X):
        return self.scalar * X


input_shape = (6, 9, 4, 20)
structure = {
    "input_shape": input_shape,
    "layers": [
        {
            "type": "split",
            "combination": "sum",
            "branches": [
                None,
                [{"type": "custom", "module": Scale, "output_shape": input_shape, "scalar": -1}],
            ],
        }
    ],
}
model = CustomNN(structure)

assert torch.all(torch.eq(model(torch.rand(input_shape)), 0)), "output should be all zeros, since this is x + (-x)"

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
