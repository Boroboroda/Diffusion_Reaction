import torch
import torch.optim
import torch.nn as nn


class Attention_PINN(nn.Module):
    def __init__(self, layers,activation=torch.nn.functional.tanh,non_negative=False):
        super(Attention_PINN, self).__init__()
        self.layers = layers
        self.activation = activation
        self.non_negative = non_negative
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        
        self.attention1 = nn.Linear(layers[0], layers[1])
        self.attention2 = nn.Linear(layers[0], layers[1])
        
        if self.activation == torch.nn.functional.silu:
            for i in range(len(layers) - 1):
                nn.init.kaiming_normal_(self.linear[i].weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                # nn.init.kaiming_uniform_(self.linear[i].weight.data, a=0, mode='fan_in', nonlinearity='sigmoid')
                nn.init.zeros_(self.linear[i].bias.data)
            
            nn.init.kaiming_normal_(self.attention1.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            # nn.init.kaiming_uniform_(self.attention1.weight.data, a=0, mode='fan_in', nonlinearity='sigmoid')
            nn.init.zeros_(self.attention1.bias.data)
            
            nn.init.kaiming_normal_(self.attention2.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            # nn.init.kaiming_uniform_(self.attention2.weight.data, a=0, mode='fan_in', nonlinearity='sigmoid')
            nn.init.zeros_(self.attention2.bias.data)
        
        elif self.activation == torch.nn.functional.tanh:
            for i in range(len(layers) - 1):
                nn.init.kaiming_normal_(self.linear[i].weight.data, nonlinearity='tanh')
                nn.init.zeros_(self.linear[i].bias.data)
            
            nn.init.kaiming_normal_(self.attention1.weight.data, nonlinearity='tanh')
            nn.init.zeros_(self.attention1.bias.data)
            
            nn.init.kaiming_normal_(self.attention2.weight.data, nonlinearity='tanh')
            nn.init.zeros_(self.attention2.bias.data)      

        elif self.activation == torch.nn.functional.relu:
            for i in range(len(layers) - 1):
                nn.init.kaiming_normal_(self.linear[i].weight.data, a=0, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(self.linear[i].bias.data)
            
            nn.init.kaiming_normal_(self.attention1.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.attention1.bias.data)
            
            nn.init.kaiming_normal_(self.attention2.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.attention2.bias.data)         
    
    """Here we introduce two encoder, to capture the feature of the input."""
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32) 
        a = self.activation(self.linear[0](x))    

        encoder_1 = self.activation(self.attention1(x))
        encoder_2 = self.activation(self.attention2(x))
        
        #Implementation of the soft-attention mechanism.
        a = a * encoder_1 + (1 - a) * encoder_2
        
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
            a = a * encoder_1 + (1 - a) * encoder_2
        
        output = self.linear[-1](a)
        if self.non_negative:
            output = torch.nn.functional.softplus(output)
        else:
            pass
        return output