import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class ShallowFCN(nn.Module):
    def __init__(self, input_size=1, n_states=2, nonlin=F.relu):
        super().__init__()
        
        self.n_states = n_states

        self.A = nn.Linear(n_states,n_states)
        self.B = nn.Linear(input_size,n_states)

        self.nonlin = nonlin

        # decreasing initial weights to increase stability
        self.A.weight = nn.Parameter(0.1 * self.A.weight)
        self.B.weight = nn.Parameter(0.1 * self.B.weight)

    def forward(self, z, x):
        y = self.A(z) + self.B(x)
        return self.nonlin(y)

class DEQModel(nn.Module):
    def __init__(self, f: nn.Module, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        z0 = torch.zeros(x.shape[0], self.f.n_states).to(x.device)

        # compute forward pass
        with torch.no_grad():
            z_star, self.forward_res = self.solver(
                lambda z : self.f(z, x),
                z0,
                **self.kwargs
            )

        # (Prepare for) Backward pass, see step 3 above
        if self.training:
            z_star.requires_grad_()
            # re-engage autograd (I believe this is necessary to compute df/dx, which is necessary for backprop)
            new_z_star = self.f(z_star, x)
            
            # Jacobian-related computations, see additional step above. For instance:
            # jac_loss = jac_loss_estimate(new_z_star, z_star, vecs=1)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()   # To avoid infinite recursion

                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                new_grad, self.backward_res = self.solver(
                    lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad,
                    torch.zeros_like(grad),
                    **self.kwargs
                )

                return new_grad

            self.hook = new_z_star.register_hook(backward_hook)
        else:
            new_z_star = z_star

        return new_z_star
