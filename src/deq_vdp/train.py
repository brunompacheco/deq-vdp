import numpy as np
import torch
import torch.nn as nn
import wandb

from deq_vdp.trainer import VdPTrainer, VdPTrainerLBFGS

def load_from_wandb(net: nn.Module, run_id: str,
                    project='van-der-pol', model_fname='model_best'):
    best_model_file = wandb.restore(
        f'{model_fname}.pth',
        run_path=f"brunompac/{project}/{run_id}",
        replace=True,
    )
    net.load_state_dict(torch.load(best_model_file.name))

    return net

if __name__ == '__main__':
    # adam_ids = ['261x5c9o', '3mqxsr82', '1lu7gxqa', '2uiugz1l', '3m364am1', '1nlxvczw', '1enfdhqu', '3skptdu1', '14w46xyu', '2xxfth8a']
    # for run_id in adam_ids:
    #     net = nn.Sequential(
    #         nn.Linear(4,20),
    #         nn.Tanh(),
    #         nn.Linear(20,20),
    #         nn.Tanh(),
    #         nn.Linear(20,20),
    #         nn.Tanh(),
    #         nn.Linear(20,20),
    #         nn.Tanh(),
    #         nn.Linear(20,2),
    #         nn.Tanh()
    #     )
    #     VdPTrainerLBFGS(
    #         load_from_wandb(net, run_id, model_fname='model_last'),
    #         epochs=100,
    #         wandb_group='FCN-LBFGS-default',
    #         random_seed=None,
    #     ).run()

    # for run_id in adam_ids:
    #     net = nn.Sequential(
    #         nn.Linear(4,20),
    #         nn.Tanh(),
    #         nn.Linear(20,20),
    #         nn.Tanh(),
    #         nn.Linear(20,20),
    #         nn.Tanh(),
    #         nn.Linear(20,20),
    #         nn.Tanh(),
    #         nn.Linear(20,2),
    #         nn.Tanh()
    #     )
    #     VdPTrainerLBFGS(
    #         load_from_wandb(net, run_id, model_fname='model_last'),
    #         epochs=2000,
    #         lbfgs_params={'max_iter': 1, 'max_eval': 3},
    #         wandb_group='FCN-LBFGS-short',
    #         random_seed=None,
    #     ).run()

    for _ in range(10):
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
        VdPTrainer(
            net,
            epochs=500,
            lr_scheduler='StepLR',
            lr_scheduler_params={
                'step_size': 100,
                'gamma': 0.5,
            },
            wandb_group='FCN-Adam-replica-3',
        ).run()
