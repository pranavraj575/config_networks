import json
import os

import torch

from config_networks import CustomNN

if __name__ == "__main__":
    fn = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "net_configs", "simple_cnn.json"
    )
    f = open(fn, "r")
    dic = json.load(f)
    f.close()
    input_shape = dic["input_shape"]
    dic["layers"].append({"type": "linear", "out_features": 1, "bias": True})

    network = CustomNN(dic)
    optim = torch.optim.Adam(network.parameters())
    c = 1
    for _ in range(100):
        optim.zero_grad()
        input = torch.rand([32] + list(input_shape))
        criterion = torch.nn.MSELoss()
        output = network(input)
        loss = criterion(output, torch.ones_like(output) * c)
        loss.backward()
        optim.step()
        print(loss)
