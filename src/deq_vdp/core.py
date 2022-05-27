import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from deq_vdp.lib.jacobian import jac_loss_estimate


def forward_iteration(f: nn.Module, x0: torch.Tensor, threshold=50, eps=1e-2):
    f0 = f(x0)
    res = []
    for k in range(threshold):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < eps):
            break

    return {"result": f0,
            "lowest": res[-1],
            "nstep": k,
            "eps": eps,
            "threshold": threshold}

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
    def __init__(self, f: nn.Module, h=lambda x: x, solver=forward_iteration, always_compute_grad=False, get_jac_loss=False, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

        self.always_compute_grad = always_compute_grad

        self.get_jac_loss = get_jac_loss

        self.h = h

    def forward(self, x: torch.Tensor):
        z0 = torch.zeros(x.shape[0], self.f.n_states).type(x.dtype).to(x.device)

        # compute forward pass
        with torch.no_grad():
            z_star_ = self.solver(
                lambda z : self.f(z, x),
                z0,
                **self.kwargs
            )['result']
        z_star = self.f(z_star_, x)

        # (Prepare for) Backward pass, see step 3 above
        if self.training or self.always_compute_grad:
            z_ = z_star.clone().detach().requires_grad_()
            # z_star.requires_grad_()
            # re-engage autograd. this is necessary to add the df/d(*) hook
            f_ = self.f(z_, x)

            # Jacobian-related computations, see additional step above. For instance:
            if self.get_jac_loss and self.training:
                jac_loss = jac_loss_estimate(f_, z_, vecs=1)

            # new_z_start already has the df/d(*) hook, but the J_g^-1 must be added mannually
            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()   # To avoid infinite recursion

                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                # forward iteration is the only solver through which I could backprop (tested with gradgradcheck)
                new_grad = forward_iteration(
                    lambda y: autograd.grad(f_, z_, y, retain_graph=True)[0] + grad,
                    torch.zeros_like(grad),
                    **self.kwargs
                )['result']

                return new_grad

            self.hook = z_star.register_hook(backward_hook)

        if self.get_jac_loss and self.training:
            return self.h(z_star), jac_loss
        else:
            return self.h(z_star)
