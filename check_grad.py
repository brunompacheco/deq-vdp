from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from deq_vdp.core import ShallowFCN, DEQModel
# from deq_vdp.solver import anderson, forward_iteration
from deq_vdp.lib.solvers import anderson, broyden

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t = torch.linspace(0,2*torch.pi,100)
y = torch.sin(t)

n_states = 5
f = ShallowFCN(input_size=1, n_states=n_states, nonlin=torch.tanh).double().to(device)
net = nn.Sequential(
    DEQModel(f, anderson, eps=1e-6, threshold=300, always_compute_grad=False,).double(),
    nn.Linear(n_states,1).double(),
).to(device)

loss = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

X = t.unsqueeze(-1).double().to(device)

# for e in tqdm(list(range(1000))):
#     net.train()
#     optimizer.zero_grad()

#     y_pred = net(X)

#     epoch_loss = loss(y_pred.view_as(y), y.double().to(device))

#     epoch_loss.backward()
#     optimizer.step()

# x_ = torch.ones(1,1) * 3 * torch.pi / 2

# x_.requires_grad_()
# net.eval()
# y_pred_ = net(x_)

# dy_pred_ = torch.autograd.grad(
#     y_pred_,
#     x_
# )[0]
# dy_pred_

X_ = torch.linspace(0,2*torch.pi,7)[1:-1].unsqueeze(-1)
X_ = X_.requires_grad_().double().to(device)

# class DEQFixedPoint(nn.Module):
#     def __init__(self, f, solver, **kwargs):
#         super().__init__()
#         self.f = f
#         self.solver = solver
#         self.kwargs = kwargs
        
#     def forward(self, x):
#         # compute forward pass and re-engage autograd tape
#         with torch.no_grad():
#             z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros(x.shape[0],self.f.n_states).to(x.device).double(), **self.kwargs)
#         z = self.f(z,x)
        
#         # set up Jacobian vector product (without additional forward calls)
#         z0 = z.clone().detach().requires_grad_()
#         f0 = self.f(z0,x)
#         def backward_hook(grad):
#             g, self.backward_res = self.solver(lambda y : torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
#                                                grad, **self.kwargs)
#             return g
                
#         z.register_hook(backward_hook)
#         return z


# n_states = 5
# f = ShallowFCN(input_size=1, n_states=n_states, nonlin=torch.tanh).double().to(device)
# # deq = DEQFixedPoint(f, anderson, tol=1e-10, max_iter=500).double().to(device)
# deq = DEQModel(f, anderson, tol=1e-10, max_iter=500).double().to(device)

# torch.autograd.gradcheck(deq, torch.randn(1,1).double().requires_grad_().to(device), eps=1e-5, atol=1e-3, check_undefined_grad=False)

torch.autograd.gradcheck(net, X_, eps=1e-5, atol=1e-3, check_undefined_grad=False, raise_exception=True)
torch.autograd.gradgradcheck(net, X_, eps=1e-5, atol=1e-3, check_undefined_grad=False, raise_exception=True)

print('success')
