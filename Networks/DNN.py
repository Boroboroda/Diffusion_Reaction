"""
@Project: Diffusion Reaction PDE
@File: DNN.py
@Application: Original PINN Neural Network
"""
import torch
import torch.optim
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers, activation=torch.nn.functional.tanh,non_negative = False):
        super(PINN, self).__init__()
        # Size of Networks
        self.layers = layers
        self.activation = activation
        self.non_nagative = non_negative

        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        if self.activation == torch.nn.functional.silu:
            for i in range(len(layers) - 1):
                nn.init.kaiming_normal_(self.linear[i].weight.data, nonlinearity='relu')
                nn.init.zeros_(self.linear[i].bias.data)
        
        elif self.activation == torch.nn.functional.tanh:
            for i in range(len(layers) - 1):
                nn.init.kaiming_normal_(self.linear[i].weight.data, nonlinearity='tanh')
                nn.init.zeros_(self.linear[i].bias.data)
        
        elif self.activation == torch.nn.functional.relu:
            for i in range(len(layers) - 1):
                nn.init.kaiming_uniform_(self.linear[i].weight.data, nonlinearity='relu')
                nn.init.zeros_(self.linear[i].bias.data)        

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)  # Ensure input is a tensor
        a = self.activation(self.linear[0](x))
        
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)       
            a = self.activation(z)
        
        # Use the softplus to constrain the output to be non-negative
        output = self.linear[-1](a)
        
        if self.non_nagative:
            output = torch.nn.functional.softplus(output)
        else:
            pass
        
        return output        
