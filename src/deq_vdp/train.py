import torch.nn as nn

from deq_vdp.trainer import VdPTrainer


if __name__ == '__main__':
    net = nn.Sequential(
        nn.Linear(4,20),
        nn.Tanh(),
        nn.Linear(20,20),
        nn.Tanh(),
        nn.Linear(20,20),
        nn.Tanh(),
        nn.Linear(20,20),
        nn.Tanh(),
        nn.Linear(20,2),
        nn.Tanh()
    )
    VdPTrainer(net, epochs=100).run()
