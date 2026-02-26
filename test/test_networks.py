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


def get_output_shape(obj, unbatch):
    if isinstance(obj, torch.Tensor):
        shape = obj.shape
        if unbatch:
            return tuple(shape[1:])
    else:
        return tuple(get_output_shape(o, unbatch) for o in obj)


def equality(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    else:
        return all(equality(ap, bp) for ap, bp in zip(a, b))


def network_equality(net1, net2):
    sd1 = net1.state_dict()
    sd2 = net2.state_dict()
    return all(torch.equal(sd1[k], sd2[k]) for k in sd1)


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


@pytest.mark.parametrize(
    "batch_size",
    batch_sizes,
)
@pytest.mark.parametrize(
    "network_file",
    [os.path.join(network_dir, fn) for fn in os.listdir(network_dir)],
)
def test_save_and_load(batch_size, network_file):
    f = open(network_file, "r")
    if network_file.endswith(".json"):
        config_dict = json.load(f)
    else:
        config_dict = ast.literal_eval(f.read())
    f.close()
    network = CustomNN(config_dict)
    network2 = CustomNN(config_dict)
    input_shape = config_dict["input_shape"]
    input = generate_random_input(batch_size, input_shape)
    network2.load_state_dict(network.state_dict())
    output = network(input)
    output2 = network2(input)
    assert equality(output, output2)
    assert network_equality(network, network2)


@pytest.mark.parametrize(
    "batch_size",
    batch_sizes,
)
@pytest.mark.parametrize(
    "network_file",
    [os.path.join(network_dir, fn) for fn in os.listdir(network_dir)],
)
def test_train(batch_size, network_file):
    f = open(network_file, "r")
    if network_file.endswith(".json"):
        config_dict = json.load(f)
    else:
        config_dict = ast.literal_eval(f.read())
    f.close()
    network = CustomNN(config_dict)
    network2 = CustomNN(config_dict)
    network2.load_state_dict(network.state_dict())
    input_shape = config_dict["input_shape"]
    # train and assert that gradients flow through all parameters
    optim = torch.optim.Adam(network2.parameters())

    def loss(t):
        if type(t) is torch.Tensor:
            return torch.mean(torch.square(t))
        else:
            return torch.sum(torch.stack([loss(tt) for tt in t]))

    for _ in range(5):
        optim.zero_grad()
        input = generate_random_input(batch_size, input_shape)
        los = loss(network2(input))
        los.backward()
        optim.step()
    input = generate_random_input(batch_size, input_shape)
    output = network(input)
    output2 = network2(input)
    assert not equality(output, output2)
    assert not network_equality(network, network2)
