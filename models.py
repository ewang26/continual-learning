import torch.nn as nn
import torch

# MNLIST_MLP_ARCH = {"sizes":[784, 256, 256, 10], "acts":['relu', 'relu']}
MNLIST_MLP_ARCH = {"sizes": [784, 1000, 10], "acts": ["relu"]}
CIFAR10_ARCH = {
    "in_channels": 3,
    "out_channels": 10,
    "l1_out_channels": 32,
    "l2_out_channels": 32,
    "l3_out_channels": 64,
    "l4_out_channels": 64,
}
# TODO CIFAR100 Should use larger network
CIFAR100_ARCH = {
    "in_channels": 3,
    "out_channels": 10,
    "l1_out_channels": 32,
    "l2_out_channels": 32,
    "l3_out_channels": 64,
    "l4_out_channels": 64,
}


class MLP(nn.Module):
    # n.b. every object of this class in classification will only return logits for each class
    # - you need to manually apply a final sigmoid activation
    def __init__(self, sizes, acts):
        if len(sizes) != len(acts) + 2:
            raise ValueError(
                f"length of sizes ({len(sizes)}) and activations ({len(acts)}) are incompatible"
            )
        super(type(self), self).__init__()
        self.num_layers = len(sizes) - 1
        lower_modules = []
        for i in range(self.num_layers - 1):
            lower_modules.append(nn.Linear(sizes[i], sizes[i + 1]))
            if acts[i] == "relu":
                lower_modules.append(nn.ReLU())
            elif acts[i] == "sigmoid":
                lower_modules.append(nn.Sigmoid())
            else:
                raise ValueError(
                    f"{acts[i]} activation layer hasn't been implemented in this code"
                )

        self.layers = nn.Sequential(*lower_modules)
        # layers are separated so that preactivations can be easily returned
        self.output_layer = nn.Linear(sizes[-2], sizes[-1])

    # if required, also return the preactivations - the result of running the data through
    # all but the last layer
    def forward(self, x, return_preactivations=False):
        pre_o = self.layers(x)
        o = self.output_layer(pre_o)
        if not return_preactivations:
            return o
        return o, pre_o
    

class CifarNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        l1_out_channels,
        l2_out_channels,
        l3_out_channels,
        l4_out_channels,
    ):
        super(type(self), self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, l1_out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(l1_out_channels, l2_out_channels, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(p=0.25),
            nn.Conv2d(l2_out_channels, l3_out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(l3_out_channels, l4_out_channels, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(p=0.25),
        )
        self.linear_block = nn.Sequential(
            nn.Linear(l4_out_channels * 6 * 6, 512), nn.ReLU()#, nn.Dropout(p=0.5)
        )
        self.out_block = nn.Linear(512, out_channels)

    def weight_init(self):
        nn.init.constant_(self.out_block.weight, 0)
        nn.init.constant_(self.out_block.bias, 0)

    def forward(self, x):
        o = self.conv_block(x)
        o = torch.flatten(o, 1)
        o = self.linear_block(o)
        o = self.out_block(o)
        return o
