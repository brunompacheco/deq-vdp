import torch
import torch.nn as nn

from deq_vdp.core import forward_iteration


class SimpleEq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z0, A_weight, A_bias, B_weight, B_bias, nonlin=torch.tanh, solver=forward_iteration):
        f = lambda x,z: nonlin(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias)

        with torch.no_grad():
            z_star = solver(
                # lambda z: nonlin(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias),
                lambda z: f(x,z),
                z0,
                threshold=200, eps=1e-4,
            )['result']

        ctx.save_for_backward(z_star.detach(), x, A_weight, A_bias, B_weight, B_bias)
        ctx.solver = solver
        ctx.nonlin = nonlin

        return z_star

    @staticmethod
    def backward(ctx, grad_output):
        z, x, A_weight, A_bias, B_weight, B_bias, = ctx.saved_tensors

        f = lambda x,z: ctx.nonlin(z @ A_weight.T + A_bias + x @ B_weight.T + B_bias)

        z.requires_grad_()
        with torch.enable_grad():
            f_ = f(x,z)

            grad_z = ctx.solver(
                lambda g: torch.autograd.grad(f_, z, g, retain_graph=True)[0] + grad_output,
                torch.zeros_like(grad_output),
                threshold=200, eps=1e-4,
            )['result']

        new_grad_x = torch.autograd.grad(f_, x, grad_z, retain_graph=True)[0]
        new_grad_A_weight = torch.autograd.grad(f_, A_weight, grad_z, retain_graph=True)[0]
        new_grad_A_bias = torch.autograd.grad(f_, A_bias, grad_z, retain_graph=True)[0]
        new_grad_B_weight = torch.autograd.grad(f_, B_weight, grad_z, retain_graph=True)[0]
        new_grad_B_bias = torch.autograd.grad(f_, B_bias, grad_z)[0]

        return new_grad_x, None, new_grad_A_weight, new_grad_A_bias, new_grad_B_weight, new_grad_B_bias, None

if __name__ == '__main__':
    batch_size = 3
    input_size = 1
    n_states = 2

    A = nn.Linear(n_states,n_states).double()
    B = nn.Linear(input_size,n_states).double()

    x = torch.rand(batch_size,input_size).double()
    x.requires_grad_()

    z0 = torch.zeros(x.shape[0], n_states).double()

    deq = SimpleEq(n_states, torch.tanh)
    # z_star = deq.apply(x, z0, A.weight, A.bias, B.weight, B.bias)

    # f = lambda x,z: F.tanh(z @ A.weight.T + A.bias + x @ B.weight.T + B.bias)
    # f_ = f(x,z_star)

    # grad_fz = torch.autograd.grad(f_.sum(), z_star)
    # print(grad_fz.shape)

    # grad_z = torch.autograd.grad(z_star.sum(), x, retain_graph=True)[0]
    # print(grad_z.shape)

    torch.autograd.gradcheck(
        lambda x: deq.apply(x, z0, A.weight, A.bias, B.weight, B.bias),
        x,
        eps=1e-4,
        atol=1e-3,
        # rtol=1e-4,
    )
    torch.autograd.gradcheck(
        lambda A_weight: deq.apply(x, z0, A_weight, A.bias, B.weight, B.bias),
        A.weight,
        eps=1e-4,
        atol=1e-3,
    )
    torch.autograd.gradcheck(
        lambda A_bias: deq.apply(x, z0, A.weight, A_bias, B.weight, B.bias),
        A.bias,
        eps=1e-4,
        atol=1e-3,
    )
    torch.autograd.gradcheck(
        lambda B_weight: deq.apply(x, z0, A.weight, A.bias, B_weight, B.bias),
        B.weight,
        eps=1e-4,
        atol=1e-3,
    )
    torch.autograd.gradcheck(
        lambda B_bias: deq.apply(x, z0, A.weight, A.bias, B.weight, B_bias),
        B.bias,
        eps=1e-4,
        atol=1e-3,
    )
    print('Gradient test passed')

    # torch.autograd.gradgradcheck(
    #     lambda x: deq.apply(x, z0, A.weight, A.bias, B.weight, B.bias),
    #     x,
    #     eps=1e-4,
    #     atol=1e-3,
    #     # rtol=1e-4,
    # )
    # torch.autograd.gradgradcheck(
    #     lambda A_weight: deq.apply(x, z0, A_weight, A.bias, B.weight, B.bias),
    #     A.weight,
    #     eps=1e-4,
    #     atol=1e-3,
    # )
    # torch.autograd.gradgradcheck(
    #     lambda A_bias: deq.apply(x, z0, A.weight, A_bias, B.weight, B.bias),
    #     A.bias,
    #     eps=1e-4,
    #     atol=1e-3,
    # )
    # torch.autograd.gradgradcheck(
    #     lambda B_weight: deq.apply(x, z0, A.weight, A.bias, B_weight, B.bias),
    #     B.weight,
    #     eps=1e-4,
    #     atol=1e-3,
    # )
    # torch.autograd.gradgradcheck(
    #     lambda B_bias: deq.apply(x, z0, A.weight, A.bias, B.weight, B_bias),
    #     B.bias,
    #     eps=1e-4,
    #     atol=1e-3,
    # )
    # print('Double gradient test passed')
