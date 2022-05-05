import torch

from torchdiffeq import odeint


def f(y, u, mu=1.):
    return torch.stack((
        y[...,1],
        mu * (1 - y[...,0]**2) * y[...,1] - y[...,0] + u[...,0]
    ),dim=-1)

def get_data(Nt=1e3, Nf=1e5, u_bounds=(-1., 1.), y_bounds=(-3., 3.)):
    Nt = int(Nt)
    Nf = int(Nf)

    u_range = u_bounds[1] - u_bounds[0]
    y_range = y_bounds[1] - y_bounds[0]

    Y_t = torch.rand(Nt,2) * y_range + y_bounds[0]  # State data points for training
    U_t = torch.rand(Nt,1) * u_range + u_bounds[0]  # Input data points for training
    X_t = torch.cat((torch.zeros(Nt,1),Y_t,U_t), dim=1)

    Y_f = torch.rand(Nf,2) * y_range + y_bounds[0]  # State data points for training
    U_f = torch.rand(Nf,1) * u_range + u_bounds[0]  # Input data points for training
    X_f = torch.cat((torch.zeros(Nf,1),Y_f,U_f), dim=1)

    # dY_f = f(Y_f,U_f)

    return (X_t, Y_t), (X_f, U_f)
