import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim


### DEFINITIONS ###

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

def anderson(f: nn.Module, x0: torch.Tensor, m=3, lam=1e-4, max_iter=50,
             tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d = x0.shape
    X = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res

### Actual Example ###

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mu = 1.
dynamics = lambda y, u: torch.stack((
    y[...,1],
    mu * (1 - y[...,0]**2) * y[...,1] - y[...,0] + u[...,0]
), dim=-1)

# data sample
t = torch.zeros(2,1).to(device)
Y = torch.Tensor([
    [-1.2256,  1.5036],
    [-0.1982,  1.6614],
]).to(device)
U = torch.Tensor([
    [-0.6281],
    [0.7932],
]).to(device)

X = torch.cat((t,Y,U), dim=-1).requires_grad_()
dY = dynamics(Y, U)

model = nn.Sequential(
    DEQModel(
        ShallowFCN(input_size=X.shape[-1], n_states=20, nonlin=torch.tanh).to(device),
        anderson, tol=1e-6, max_iter=500,
    ),
    nn.Linear(20,2),
    nn.Tanh(),
).to(device)
h = lambda y: y * 3

lamb = .05
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
model.train()
optimizer.zero_grad()
with torch.autograd.set_detect_anomaly(True):
    Y_pred = h(model(X))

    # get dY_pred / dt
    dY0_pred = autograd.grad(
        Y_pred[:,0],
        X,
        grad_outputs=torch.ones_like(Y_pred[:,0]),
        # retain_graph=True,
        create_graph=False
    )[0][:,0]  # dY_0 / dt

    dY1_pred = autograd.grad(
        Y_pred[:,1],
        X,
        grad_outputs=torch.ones_like(Y_pred[:,1]),
        # retain_graph=False,
        create_graph=False
    )[0][:,0]  # dY_1 / dt

    dY_pred = torch.stack((dY0_pred, dY1_pred), dim=-1)

    loss = criterion(dY_pred, dY)
    loss.backward()

    optimizer.step()
