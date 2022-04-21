import torch
import torch.nn as nn
import torch.autograd as autograd


class ShallowFCN(nn.Module):
    def __init__(self, input_size=1, n_states=2):
        super().__init__()
        
        self.n_states = n_states

        self.A = nn.Linear(n_states,n_states)
        self.B = nn.Linear(input_size,n_states)

        # decreasing initial weights to increase stability
        self.A.weight = nn.Parameter(0.1 * self.A.weight)
        self.B.weight = nn.Parameter(0.1 * self.B.weight)

    def forward(self, z, x):
        return torch.sigmoid(self.A(z) + self.B(x))

class DEQFixedPoint(nn.Module):
    def __init__(self, f: nn.Module, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        # compute forward pass
        with torch.no_grad():
            z, self.forward_res = self.solver(
                lambda z : self.f(z, x),
                torch.zeros(x.shape[0], self.f.n_states),
                **self.kwargs
            )

        # re-engage autograd, like a forward pass with a really good initial guess
        z = self.f(z,x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)
        def backward_hook(y):
            g, self.backward_res = self.solver(
                lambda g : autograd.grad(f0, z0, g, retain_graph=True)[0] + y,
                y,
                **self.kwargs
            )
            return g

        z.register_hook(backward_hook)
        return z
