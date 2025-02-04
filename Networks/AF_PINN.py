import torch
import torch.nn as nn
import math

class AF_PINN(nn.Module):
    def __init__(self, layers, activation=torch.nn.functional.tanh, non_negative=False,
                 use_rff=False, rff_num_features=None, rff_sigma=1.0):
        """
        Physics-Informed Neural Network with optional Random Fourier Features

        Args:
            layers (list): Network layer dimensions
            activation (function): Activation function to use
            non_negative (bool): Whether to constrain output to non-negative values
            use_rff (bool): Whether to use Random Fourier Features
            rff_num_features (int): Number of random Fourier features to use
            rff_sigma (float): Bandwidth parameter for RFF
        """
        super(AF_PINN, self).__init__()

        # Network configuration
        self.layers = layers.copy()  # Create a copy to avoid modifying the original
        self.activation = activation
        self.non_negative = non_negative
        self.use_rff = use_rff

        # Random Fourier Features setup
        if self.use_rff:
            # Get input dimension from first layer
            input_dim = layers[0]

            # If rff_num_features is not specified, default to input dimension
            if rff_num_features is None:
                rff_num_features = input_dim

            # RFF weights and bias
            self.rff_weights = nn.Parameter(
                torch.normal(0, 1 / rff_sigma, size=(input_dim, rff_num_features))
            )
            self.rff_bias = nn.Parameter(
                2 * math.pi * torch.rand(rff_num_features)
            )

            # Modify first layer input size to accommodate RFF
            # We double the number of features due to cos and sin projections
            layers[0] = 2 * rff_num_features

        # Linear layers
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        self.attention1 = nn.Linear(layers[0], layers[1])
        self.attention2 = nn.Linear(layers[0], layers[1])

        # Weight initialization based on activation function
        self._init_weights(activation)

    def _init_weights(self, activation):
        """Initialize weights based on activation function"""
        if activation == torch.nn.functional.silu:
            init_method = lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')
        elif activation == torch.nn.functional.tanh:
            init_method = lambda w: nn.init.kaiming_normal_(w, nonlinearity='tanh')
        elif activation == torch.nn.functional.relu:
            init_method = lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu')
        else:
            init_method = nn.init.xavier_uniform_

        for i in range(len(self.layers) - 1):
            init_method(self.linear[i].weight.data)
            nn.init.zeros_(self.linear[i].bias.data)

        nn.init.kaiming_normal_(self.attention1.weight.data, nonlinearity='tanh')
        nn.init.zeros_(self.attention1.bias.data)

        nn.init.kaiming_normal_(self.attention2.weight.data, nonlinearity='tanh')
        nn.init.zeros_(self.attention2.bias.data)

    def random_fourier_features(self, x):
        """
        Apply Random Fourier Features mapping

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: RFF transformed input
        """
        # Ensure x is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Linear transformation
        x_proj = torch.matmul(x, self.rff_weights) + self.rff_bias

        # Compute random features
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

    def forward(self, x):
        # Ensure input is a tensor
        x = torch.as_tensor(x, dtype=torch.float32)

        # Apply Random Fourier Features if enabled
        if self.use_rff:
            x = self.random_fourier_features(x)

        # First layer
        a = self.activation(self.linear[0](x))

        encoder_1 = self.activation(self.attention1(x))
        encoder_2 = self.activation(self.attention2(x))

        # Implementation of the soft-attention mechanism.
        a = a * encoder_1 + (1 - a) * encoder_2
        # Hidden layers
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
            a = a * encoder_1 + (1 - a) * encoder_2

        # Output layer
        output = self.linear[-1](a)

        # Non-negativity constraint
        if self.non_negative:
            output = torch.nn.functional.softplus(output)

        return output
