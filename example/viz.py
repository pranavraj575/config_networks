from config_networks import CustomNN
import json,os,torchviz,torch
DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(DIR,'net_configs','simple_cnn.json')) as f:
    structrue=json.load(f)
net=CustomNN(structrue)

x=torch.rand(structrue['input_shape']).unsqueeze(0)
y=net(x)
print(net)
print(y.shape,net.output_shape)
torchviz.make_dot(y,params=dict(list(net.named_parameters()))).render("rnn_torchviz", format="png")