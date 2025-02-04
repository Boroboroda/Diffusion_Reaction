import torch

def gradients(u, x, order = 1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                        create_graph=True,
                                        only_inputs=True)[0]
    return gradients(gradients(u, x), x, order= order - 1)