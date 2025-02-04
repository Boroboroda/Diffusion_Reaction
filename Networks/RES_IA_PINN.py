import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, mid, activation):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(mid, mid)
        self.activation = activation
        self.linear2 = nn.Linear(mid, mid)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out += residual  # 残差连接
        return out

# 改进的Attention_PINN类，加入ResBlock
class RES_IA_PINN(nn.Module):
    def __init__(self, layers, activation=torch.nn.functional.tanh, non_negative=False):
        super(RES_IA_PINN, self).__init__()
        self.layers = layers
        self.activation = activation
        self.non_negative = non_negative

        # 定义线性层
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        # 定义注意力层
        self.attention1 = nn.Linear(layers[0], layers[1])
        self.attention2 = nn.Linear(layers[0], layers[1])

        # 添加多个残差块
        self.res_blocks = nn.ModuleList([ResBlock(layers[i + 1], self.activation) for i in range(len(layers) - 2)])

        # 初始化参数
        self.init_weights()

    def init_weights(self):
        for layer in self.linear:
            if self.activation == torch.nn.functional.silu:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='sigmoid')
            elif self.activation == torch.nn.functional.tanh:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='tanh')
            elif self.activation == torch.nn.functional.relu:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')                
            nn.init.zeros_(layer.bias)
        
        # 初始化注意力层
        if self.activation == torch.nn.functional.silu:
            nn.init.kaiming_uniform_(self.attention1.weight, nonlinearity='sigmoid')
            nn.init.kaiming_uniform_(self.attention2.weight, nonlinearity='sigmoid')
        
        elif self.activation == torch.nn.functional.tanh:
            nn.init.kaiming_uniform_(self.attention1.weight, nonlinearity='tanh')
            nn.init.kaiming_uniform_(self.attention1.weight, nonlinearity='tanh')

        elif self.activation == torch.nn.functional.relu:
            nn.init.kaiming_uniform_(self.attention1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.attention1.weight, nonlinearity='relu')        
        
        nn.init.zeros_(self.attention1.bias)
        nn.init.zeros_(self.attention2.bias)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)  # 确保输入是张量
        a = self.activation(self.linear[0](x))  # 初始线性层

        # 两个编码器生成注意力权重
        encoder_1 = self.activation(self.attention1(x))
        encoder_2 = self.activation(self.attention2(x))
        
        # Soft-attention机制
        a = a * encoder_1 + (1 - a) * encoder_2

        # 通过多个残差块
        # 每层隐藏层之间加入残差块
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)  # 线性层
            a = self.activation(z)
            a = self.res_blocks[i - 1](a)  # 添加残差块
            a = a * encoder_1 + (1 - a) * encoder_2  # Attention机制
        
        # 输出层
        output = self.linear[-1](a)
        
        # 如果需要非负输出，使用ReLU
        if self.non_negative:
            output = torch.nn.functional.softplus(output)
        else:
            pass
        return output