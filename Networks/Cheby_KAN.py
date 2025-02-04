import torch
import torch.nn.functional as F
import math
import random

"""Set seed"""


def seed_torch(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_torch(42)


class ChebyKANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            degree=5,
            scale_base=1.0,
            scale_cheby=1.0,
            base_activation=torch.nn.SiLU,
            use_bias=True,
    ):
        """
        Initial ChebyKANLinear Layer.

        Parameters:
            in_features (int): Input dim.
            out_features (int): Output dim.
            degree (int): Chebyshev Polynomial order.
            scale_base (float): The initialization of basic weight.
            scale_cheby (float): Chebyshev factor for coefficient initialization.
            base_activation (nn.Module): /
            use_bias (bool): /
        """
        super(ChebyKANLinear, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features  
        self.degree = degree 
        self.scale_base = scale_base  
        self.scale_cheby = scale_cheby  
        self.base_activation = base_activation()  
        self.use_bias = use_bias  

        # Initialize the basic weight parameters, the shape is (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # Initialize chebyshev coefficient parameters, the shape is (out_features, in_features, degree + 1)
        self.cheby_coeffs = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, degree + 1)
        )

        if self.use_bias:
            # Initialized bias item, the shape is (out_features,)
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        # Precomputing the order index of a Chebyshev polynomial of the shape (degree + 1,)
        self.register_buffer("cheby_orders", torch.arange(0, degree + 1))

        self.reset_parameters()

    def reset_parameters(self):
        # Use Kaiming to initialize base_weight
        torch.nn.init.kaiming_normal_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # Normal Distribution -  Chebyshev parameter cheby_coeffs
        with torch.no_grad():
            std = self.scale_cheby / math.sqrt(self.in_features)
            self.cheby_coeffs.normal_(mean=0.0, std=std)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.normal_(self.bias, mean=0.0, std=bound)

    # def reset_parameters(self):
    #     # 使用 Xavier 初始化基础权重参数 base_weight
    #     torch.nn.init.xavier_normal_(self.base_weight, gain=self.scale_base)

    #     # 使用正态分布初始化 Chebyshev 系数参数 cheby_coeffs
    #     with torch.no_grad():
    #         std = self.scale_cheby / math.sqrt(self.in_features)
    #         self.cheby_coeffs.normal_(mean=0.0, std=std)

    #     if self.use_bias:
    #         fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         torch.nn.init.normal_(self.bias, mean=0.0, std=bound)

    def chebyshev_polynomials(self, x: torch.Tensor):
        """t
        Caculate Chebyshev-Polynomial value of x.

        Parameter:aculate
            x (torch.Tensor): Input in shape (batch_size, in_features)

        返回:
            torch.Tensor: Chebyshev polynomial value in shape (batch_size, in_features, degree + 1)
        """
        # scaling x into [-1, 1].
        x = torch.tanh(x)

        # caculate arccos(x), to get Chebyshev-value
        theta = torch.acos(x)  # In shape (batch_size, in_features)

        # cheby_orders in shape (degree + 1,)
        # theta.unsqueeze(-1) in shape (batch_size, in_features, 1)
        # Caculate theta * n,in shape (batch_size, in_features, degree + 1)
        theta_n = theta.unsqueeze(-1) * self.cheby_orders

        # Caculate cos(n * arccos(x)),get  Chebyshev -value
        T_n = torch.cos(theta_n)  # in shape (batch_size, in_features, degree + 1)

        return T_n

    def forward(self, x: torch.Tensor):
        """
        Forward propagation

        Parameter:
            x (torch.Tensor): Input,in shape (..., in_features).

        Return:
            torch.Tensor: Output,in shape (..., out_features).
        """
        original_shape = x.shape

        # Flatten the input into a two-dimensional tensor, in shape (-1, in_features)
        x = x.view(-1, self.in_features)

        # Caculate linear transformation
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Caculate Chebyshev 
        T_n = self.chebyshev_polynomials(x)  # in shape (batch_size, in_features, degree + 1)

        # Caculate Chebyshev 
        cheby_output = torch.einsum('bik,oik->bo', T_n, self.cheby_coeffs)

        # Combining Chebyshev output
        output = base_output + cheby_output

        # Add bias
        if self.use_bias:
            output += self.bias

        # Restore the shape of the output tensor
        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        Caculate Chebyshev regularization loss

        Parameter:
            regularize_coeffs (float): The regularization coefficient.

        Return:
            torch.Tensor: Regularized loss values.
        """
        # Caculate Chebyshev L2 norm
        coeffs_l2 = self.cheby_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2


class ChebyKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            degree=5,
            scale_base=1.0,
            scale_cheby=1.0,
            base_activation=torch.nn.SiLU,
            use_bias=True,
            use_layer_norm=True,  # 添加LayerNorm的控制参数
            layer_norm_eps=1e-5,  # LayerNorm的epsilon参数
            non_negative=False,
    ):
        """
        Initial ChebyKAN Model。

        Parameter:
            layers_hidden (list): /
            degree (int): Chebyshev /
            scale_base (float): /
            scale_cheby (float): /
            base_activation /
            use_bias (bool): /
            use_layer_norm (bool): Use Layernorm to effectively avoid the gradient problem of shrinking to the [1, -1] interval.
            layer_norm_eps (float): /
        """
        super(ChebyKAN, self).__init__()

        self.use_layer_norm = use_layer_norm
        self.non_negative = non_negative

        # Initialization
        self.layers = torch.nn.ModuleList()
        if use_layer_norm:
            self.layer_norms = torch.nn.ModuleList()

        # Structure
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                ChebyKANLinear(
                    in_features,
                    out_features,
                    degree=degree,
                    scale_base=scale_base,
                    scale_cheby=scale_cheby,
                    base_activation=base_activation,
                    use_bias=use_bias,
                )
            )

            #Add LayerNorm
            if use_layer_norm:
                self.layer_norms.append(
                    torch.nn.LayerNorm(out_features, eps=layer_norm_eps)
                )

    def forward(self, x: torch.Tensor):
        """
        Forward propagation

        Parameter:
            x (torch.Tensor): in shape (..., in_features)。

        Return:
            torch.Tensor: in shape (..., out_features)。
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply LayerNorm
            if self.use_layer_norm and i < len(self.layers) - 1:
                x = self.layer_norms[i](x)
        
        output = x
        
        # Non-negative layer.
        if self.non_negative:
            output = F.softplus(x)
        return output


