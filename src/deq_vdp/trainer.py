import logging
from pathlib import Path
import random
from tabnanny import check

from time import time
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from torchdiffeq import odeint

import wandb

from deq_vdp.vdp import get_data, f


def timeit(fun):
    def fun_(*args, **kwargs):
        start_time = time()
        f_ret = fun(*args, **kwargs)
        end_time = time()

        return end_time - start_time, f_ret

    return fun_

class VdPTrainer():
    """Generic trainer for PyTorch NNs.

    Attributes:
        net: the neural network to be trained.
        epochs: number of epochs to train the network.
        lr: learning rate.
        optimizer: optimizer (name of a optimizer inside `torch.optim`).
        loss_func: a valid PyTorch loss function.
        lr_scheduler: if a scheduler is to be used, provide the name of a valid
        `torch.optim.lr_scheduler`.
        lr_scheduler_params: parameters of selected `lr_scheduler`.
        batch_size: batch_size for training.
        device: see `torch.device`.
        wandb_project: W&B project where to log and store model.
        logger: see `logging`.
        random_seed: if not None (default = 42), fixes randomness for Python,
        NumPy as PyTorch (makes trainig reproducible).
    """
    def __init__(self, net: nn.Module, Nt=1e3, Nf=1e5, u_bounds=(-1, 1),
                 y_bounds=(-3, 3), epochs=5, lr= 0.1, lamb=.1, h=None,
                 optimizer: str = 'Adam', optimizer_params: dict = None,
                 loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 lr_scheduler_params: dict = None, batch_size=16,
                 device=None, wandb_project="van-der-pol", wandb_group=None,
                 logger=None, checkpoint_every=50, random_seed=42) -> None:
        self._is_initalized = False

        self.Nt = Nt
        self.Nf = Nf
        self.u_bounds = u_bounds
        self.y_bounds = y_bounds

        # self.h = lambda z: z * self.y_bounds[1] + self.y_bounds[0]
        self.h = lambda z: z * (self.y_bounds[1] - self.y_bounds[0]) / 2

        self.lamb = lamb

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        # TODO: expose `val_params`
        y0 = torch.zeros(1,2).to(self.device)
        y0[0,1] = .1
        u = torch.zeros(1,1).to(self.device)
        dt = torch.ones(1,1).to(self.device) * 0.1
        K = 200
        Y_ref = odeint(lambda t, y: f(y,u), y0, torch.Tensor([i * dt for i in range(K+1)]), method='rk4')
        self.val_params = {
            'y0': y0,
            'u': u,
            'dt': dt,
            'K': K,
            'Y_ref': Y_ref,
        }

        self._e = 0  # inital epoch

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss_func = loss_func
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.l = logging.getLogger(__name__)
        else:
            self.l = logger

        self.checkpoint_every = checkpoint_every

        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.best_val = float('inf')

        self._log_to_wandb = False if wandb_project is None else True
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group

    def get_dY_pred(self, Y_pred, X):
        delY0_pred = torch.autograd.grad(
            Y_pred[:,0],
            X,
            grad_outputs=torch.ones_like(Y_pred[:,0]),
            create_graph=True
        )[0][:,0]  # time derivative (first input dimension)

        delY1_pred = torch.autograd.grad(
            Y_pred[:,1],
            X,
            grad_outputs=torch.ones_like(Y_pred[:,1]),
            create_graph=True
        )[0][:,0]  # time derivative (first input dimension)

        return torch.stack((delY0_pred, delY1_pred), dim=-1)

    @classmethod
    def load_trainer(cls, net: nn.Module, run_id: str, wandb_project="van-der-pol",
                     logger=None):
        """Load a previously initialized trainer from wandb.

        Loads checkpoint from wandb and create the instance.

        Args:
            run_id: valid wandb id.
            logger: same as the attribute.
        """
        wandb.init(
            project=wandb_project,
            entity="brunompac",
            id=run_id,
            resume='must',
        )

        # load checkpoint file
        checkpoint_file = wandb.restore('checkpoint.tar')
        checkpoint = torch.load(checkpoint_file.name)

        # load model
        net = net.to(wandb.config['device'])
        net.load_state_dict(checkpoint['model_state_dict'])

        # fix for older versions
        if 'lr_scheduler' not in wandb.config.keys():
            wandb.config['lr_scheduler'] = None
            wandb.config['lr_scheduler_params'] = None

        # create trainer instance
        self = cls(
            epochs=wandb.config['epochs'],
            net=net,
            lr=wandb.config['learning_rate'],
            optimizer=wandb.config['optimizer'],
            loss_func=wandb.config['loss_func'],
            lr_scheduler=wandb.config['lr_scheduler'],
            lr_scheduler_params=wandb.config['lr_scheduler_params'],
            batch_size=wandb.config['batch_size'],
            device=wandb.config['device'],
            logger=logger,
            wandb_project=wandb_project,
            random_seed=wandb.config['random_seed'],
        )

        if 'best_val' in checkpoint.keys():
            self.best_val = checkpoint['best_val']

        self._e = checkpoint['epoch'] + 1

        self.l.info(f'Resuming training of {wandb.run.name} at epoch {self._e}')

        # load optimizer
        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr,
            **self.optimizer_params
        )
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])

        # load scheduler
        if self.lr_scheduler is not None:
            Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)
            self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self._loss_func_y = eval(f"nn.{self.loss_func}()")
        self._loss_func_F = eval(f"nn.{self.loss_func}()")

        self.prepare_data()

        self._is_initalized = True

        return self

    def setup_training(self):
        self.l.info('Setting up training')

        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr
        )

        if self.lr_scheduler is not None:
            Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)

        self._loss_func_y = eval(f"nn.{self.loss_func}()")
        self._loss_func_F = eval(f"nn.{self.loss_func}()")

        if self._log_to_wandb:
            self.l.info('Initializing wandb.')
            self.initialize_wandb()

        self.l.info('Preparing data')
        self.prepare_data()

        self._is_initalized = True

    def initialize_wandb(self):
        wandb.init(
            project=self.wandb_project,
            entity="brunompac",
            group=self.wandb_group,
            config={
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "model": type(self.net).__name__,
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_params": self.lr_scheduler_params,
                "loss_func": self.loss_func,
                "random_seed": self.random_seed,
                "device": self.device,
            },
        )

        wandb.watch(self.net)

        self._id = wandb.run.id

        self.l.info(f"Wandb set up. Run ID: {self._id}")

    def prepare_data(self):
        (X_t, Y_t), (X_f, dY_f) = get_data(self.Nt, self.Nf, self.u_bounds, self.y_bounds)

        X_t = X_t.to(self.device)
        Y_t = Y_t.to(self.device)
        X_f = X_f.to(self.device)
        dY_f = dY_f.to(self.device)

        self.data = (X_t, Y_t), (X_f, dY_f)

    def run(self):
        if not self._is_initalized:
            self.setup_training()

        scaler = GradScaler()
        while self._e < self.epochs:
            self.l.info(f"Epoch {self._e} started ({self._e+1}/{self.epochs})")
            epoch_start_time = time()

            # train
            train_time, (train_loss_y, train_loss_f) = timeit(self.train_pass)(scaler)
            train_loss = train_loss_y + self.lamb * train_loss_f

            self.l.info(f"Training pass took {train_time:.3f} seconds")
            self.l.info(f"Training loss = {train_loss}")

            # validation
            val_time, iae = timeit(self.validation_pass)()

            self.l.info(f"Validation pass took {val_time:.3f} seconds")
            self.l.info(f"IAE = {iae}")

            if self._log_to_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "train_loss_y": train_loss_y,
                    "train_loss_f": train_loss_f,
                    "iae": iae,
                }, step=self._e, commit=True)

                if self._e % self.checkpoint_every == self.checkpoint_every - 1:
                    self.l.info(f"Saving checkpoint")
                    self.save_checkpoint()

            # TODO: save on moving average improvement! IAE is very unstable
            if iae < self.best_val:
                if self._log_to_wandb:
                    self.l.info(f"Saving best model")
                    self.save_model(name='model_best')

                self.best_val = iae

            epoch_end_time = time()
            self.l.info(
                f"Epoch {self._e} finished and took "
                f"{epoch_end_time - epoch_start_time:.2f} seconds"
            )

            self._e += 1

        if self._log_to_wandb:
            self.l.info(f"Saving model")
            self.save_model(name='model_last')

            wandb.finish()

        self.l.info('Training finished!')

    def train_pass(self, scaler):
        self.net.train()

        (X_t, Y_t), (X_f, dY_f) = self.data
        X_f.requires_grad_()
        with torch.set_grad_enabled(True):
            self._optim.zero_grad()

            with autocast():
                Y_t_pred = self.h(self.net(X_t))

                loss_y = self._loss_func_y(Y_t_pred, Y_t)

                Y_f_pred = self.h(self.net(X_f))

                dY_pred = self.get_dY_pred(Y_f_pred, X_f)

                loss_f = self._loss_func_F(dY_pred, dY_f)

                loss = loss_y + self.lamb * loss_f

            scaler.scale(loss).backward()

            scaler.step(self._optim)
            scaler.update()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        return loss_y.item(), loss_f.item()

    def validation_pass(self):
        self.net.eval()

        dt, y0, u, Y_ref = self.val_params['dt'],self.val_params['y0'],self.val_params['u'],self.val_params['Y_ref']
        Y = [self.val_params['y0'].cpu().detach().numpy().squeeze(),]

        x = torch.cat((dt,y0,u), dim=-1).to(self.device)
        with torch.set_grad_enabled(False):
            for _ in range(self.val_params['K']):
                y_next = self.h(self.net(x))

                Y.append(y_next.cpu().detach().numpy().squeeze())

                x = torch.cat((dt,y_next,u), dim=-1)

        iae = np.abs(Y_ref.cpu().detach().numpy() - Y).mean()

        return iae

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self._e,
            'best_val': self.best_val,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self._optim.state_dict(),
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._scheduler.state_dict()

        torch.save(checkpoint, Path(wandb.run.dir)/'checkpoint.tar')
        wandb.save('checkpoint.tar')

    def save_model(self, name='model'):
        fname = f"{name}.pth"
        fpath = Path(wandb.run.dir)/fname

        torch.save(self.net.state_dict(), fpath)
        wandb.save(fname)

        return fpath

class VdPTrainerLBFGS(VdPTrainer):
    def __init__(self, net: nn.Module, Nt=1e3, Nf=1e5, u_bounds=(-1, 1),
                 y_bounds=(-3, 3), epochs=5, lr=1., lamb=.1, h=None,
                 lbfgs_params: dict = None, loss_func: str = 'MSELoss',
                 lr_scheduler: str = None, lr_scheduler_params: dict = None,
                 batch_size=16, device=None, wandb_project="van-der-pol",
                 wandb_group=None, logger=None, checkpoint_every=50,
                 random_seed=42) -> None:
        super().__init__(net, Nt, Nf, u_bounds, y_bounds, epochs, lr, lamb, h, 'LBFGS',
                         lbfgs_params, loss_func, lr_scheduler, lr_scheduler_params,
                         batch_size, device, wandb_project, wandb_group, logger,
                         checkpoint_every, random_seed)

    def train_pass(self, scaler):
        self.net.train()

        (X_t, Y_t), (X_f, dY_f) = self.data
        X_f.requires_grad_()

        y_losses = list()
        f_losses = list()
        losses = list()
        with torch.set_grad_enabled(True):
            def closure():
                self._optim.zero_grad()

                Y_t_pred = self.h(self.net(X_t))

                loss_y = self._loss_func_y(Y_t_pred, Y_t)

                Y_f_pred = self.h(self.net(X_f))

                dY_pred = self.get_dY_pred(Y_f_pred, X_f)

                loss_f = self._loss_func_F(dY_pred, dY_f)

                loss = loss_y + self.lamb * loss_f

                y_losses.append(loss_y.item())
                f_losses.append(loss_f.item())
                losses.append(loss.item())

                if loss.requires_grad:
                    loss.backward()

                return loss

            self._optim.step(closure)

            y_losses = np.array(y_losses)
            f_losses = np.array(f_losses)
            losses = np.array(losses)

            if self.lr_scheduler is not None:
                self._scheduler.step()

        return y_losses[-1], f_losses[-1]