import torch
import torch.nn as nn


def forward_iteration(f: nn.Module, x0: torch.Tensor, max_iter=50, tol=1e-2):
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res

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
        H_ = H.clone()
        H_[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        H = H_
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]   # (bsz x n)
        
        X_new = X.clone()
        F_new = F.clone()
        X_new[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F_new[:,k%m] = f(X_new[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F_new[:,k%m] - X_new[:,k%m]).norm().item()/(1e-5 + F_new[:,k%m].norm().item()))

        X, F = X_new, F_new

        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res
