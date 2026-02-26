import ast
import json
import os

import pytest
import torch

from src.config_networks import CustomNN

batch_sizes = [1, 2, 4, 16]
network_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "net_configs"
)


def generate_random_input(batch_size, input_shape):
    if type(input_shape[0]) is int:
        return torch.normal(0, 1, (batch_size,) + tuple(input_shape))
    else:
        return tuple(generate_random_input(batch_size, s) for s in input_shape)


def get_output_shape(object, unbatch):
    if isinstance(object, torch.Tensor):
        shape = object.shape
        if unbatch:
            return tuple(shape[1:])
    else:
        return tuple(get_output_shape(o, unbatch) for o in object)


@pytest.mark.parametrize(
    "batch_size",
    batch_sizes,
)
@pytest.mark.parametrize(
    "network_file",
    [os.path.join(network_dir, fn) for fn in os.listdir(network_dir)],
)
def test_networks(batch_size, network_file):

    f = open(network_file, "r")
    if network_file.endswith(".json"):
        config_dict = json.load(f)
    else:
        config_dict = ast.literal_eval(f.read())
    f.close()
    network = CustomNN(config_dict)
    input_shape = config_dict["input_shape"]
    output = network(generate_random_input(batch_size, input_shape))
    assert get_output_shape(output, unbatch=True) == network.output_shape
