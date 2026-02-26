import pytest
import torch

from src.config_networks import layer_from_config_dict

shapes = [
    [1, 2, 3],
    [3],
    [4, 5, 6],
    [7, 9],
    [1],
]

img_shapes = [[5, 10, 10], [1, 360, 400]]
batch_sizes = [1, 2, 16]
output_features = [1, 16]
kernel_sizes = [2, [2, 3]]
paddings = [0, 1, [0, 1]]
strides = [1, 4, [2, 3]]


@pytest.mark.parametrize(
    "type_",
    ["CNN", "maxpool", "avgpool"],
)
@pytest.mark.parametrize(
    "batch_size",
    batch_sizes,
)
@pytest.mark.parametrize(
    "img_shape",
    img_shapes,
)
@pytest.mark.parametrize(
    "output_channels",
    output_features,
)
@pytest.mark.parametrize(
    "kernel_size",
    kernel_sizes,
)
@pytest.mark.parametrize(
    "stride",
    strides,
)
@pytest.mark.parametrize(
    "padding",
    paddings,
)
def test_cnn(
    type_, batch_size, img_shape, kernel_size, stride, padding, output_channels
):
    cnn_layer, output_shape = layer_from_config_dict(
        dic={
            "type": type_,
            "out_channels": output_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
        },
        input_shape=img_shape,
    )
    input = torch.rand([batch_size] + list(img_shape))

    output = cnn_layer(input)
    assert tuple(output.shape) == tuple([batch_size] + list(output_shape))


@pytest.mark.parametrize(
    "type_",
    [
        "identity",
        "relu",
        "sigmoid",
        "leakyrelu",
        "tanh",
        "softmax",
        "dropout",
    ],
)
@pytest.mark.parametrize(
    "shape",
    shapes,
)
def test_easy(type_, shape):
    input = torch.normal(0, 1, size=shape)
    extra_args = dict()
    if type_ == "softmax":
        extra_args = {"dim": -1}
    layer, output_shape = layer_from_config_dict(
        dic={
            "type": type_,
        }
        | extra_args,
        input_shape=shape,
    )

    output = layer(input)
    assert tuple(output.shape) == tuple(output_shape)
    if type_ == "relu":
        assert torch.all(output >= 0)
        pos = torch.where(input >= 0)
        assert torch.all(input[pos] == output[pos])


@pytest.mark.parametrize(
    "shape",
    shapes,
)
@pytest.mark.parametrize(
    "out_features",
    output_features,
)
def test_linear(shape, out_features):
    input = torch.normal(0, 1, size=shape)
    layer, output_shape = layer_from_config_dict(
        dic={"type": "linear", "out_features": out_features},
        input_shape=shape,
    )

    output = layer(input)
    assert tuple(output.shape) == tuple(output_shape)


@pytest.mark.parametrize(
    "shape",
    shapes,
)
@pytest.mark.parametrize(
    "batch_size",
    batch_sizes,
)
def test_flatten(shape, batch_size):
    layer, output_shape = layer_from_config_dict(
        dic={"type": "flatten"},
        input_shape=shape,
    )
    input = torch.normal(0, 1, size=[batch_size] + list(shape))
    output = layer(input)
    assert tuple(output.shape) == tuple([batch_size] + list(output_shape))
