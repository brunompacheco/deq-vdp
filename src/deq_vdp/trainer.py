from abc import ABC, abstractmethod

import logging
from pathlib import Path
import random

from time import time
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler

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

class Trainer(ABC):
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
        device: see `torch.device`.
        wandb_project: W&B project where to log and store model.
        logger: see `logging`.
        random_seed: if not None (default = 42), fixes randomness for Python,
        NumPy as PyTorch (makes trainig reproducible).
    """
    def __init__(self, net: nn.Module, epochs=5, lr= 0.1,
                 optimizer: str = 'Adam', optimizer_params: dict = None,
                 loss_func: str = 'MSELoss', lr_scheduler: str = None,
                 lr_scheduler_params: dict = None,
                 device=None, wandb_project=None, wandb_group=None,
                 logger=None, checkpoint_every=50, random_seed=42) -> None:
        self._is_initalized = False

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self._e = 0  # inital epoch

        self.epochs = epochs
        self.lr = lr

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

        self._loss_func = eval(f"nn.{self.loss_func}()")

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

        if self._log_to_wandb:
            if not hasattr(self, '_wandb_config'):
                self._wandb_config = dict()

            for k, v in {
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "model": type(self.net).__name__,
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_params": self.lr_scheduler_params,
                "loss_func": self.loss_func,
                "random_seed": self.random_seed,
                "device": self.device,
            }.items():
                self._wandb_config[k] = v

            self.l.info('Initializing wandb.')
            self.initialize_wandb()

        self._loss_func = eval(f"nn.{self.loss_func}()")

        self.l.info('Preparing data')
        self.prepare_data()

        self._is_initalized = True

    def initialize_wandb(self):
        wandb.init(
            project=self.wandb_project,
            entity="brunompac",
            group=self.wandb_group,
            config=self._wandb_config,
        )

        wandb.watch(self.net)

        self._id = wandb.run.id

        self.l.info(f"Wandb set up. Run ID: {self._id}")

    @abstractmethod
    def prepare_data(self):
        """Must populate `self.data` and `self.val_data`.
        """

    def _run_epoch(self):
        # train
        train_time, train_loss = timeit(self.train_pass)()

        self.l.info(f"Training pass took {train_time:.3f} seconds")
        self.l.info(f"Training loss = {train_loss}")

        # validation
        val_time, val_loss = timeit(self.validation_pass)()

        self.l.info(f"Validation pass took {val_time:.3f} seconds")
        self.l.info(f"Validation loss = {val_loss}")

        data_to_log = {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        val_score = val_loss  # defines best model

        return data_to_log, val_score

    def run(self):
        if not self._is_initalized:
            self.setup_training()

        self._scaler = GradScaler()
        while self._e < self.epochs:
            self.l.info(f"Epoch {self._e} started ({self._e+1}/{self.epochs})")
            epoch_start_time = time()

            data_to_log, val_score = self._run_epoch()

            if self._log_to_wandb:
                wandb.log(data_to_log, step=self._e, commit=True)

                if self._e % self.checkpoint_every == self.checkpoint_every - 1:
                    self.l.info(f"Saving checkpoint")
                    self.save_checkpoint()

            if val_score < self.best_val:
                if self._log_to_wandb:
                    self.l.info(f"Saving best model")
                    self.save_model(name='model_best')

                self.best_val = val_score

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

    def train_pass(self):
        train_loss = 0
        self.net.train()
        with torch.set_grad_enabled(True):
            for X, y in self.data:
                X = X.to(self.device)
                y = y.to(self.device)

                self._optim.zero_grad()

                with autocast():
                    y_hat = self.net(X)

                    loss = self._loss_func(y_hat, y)

                self._scaler.scale(loss).backward()

                train_loss += loss.item() * len(y)

                self._scaler.step(self._optim)
                self._scaler.update()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        # scale to data size
        train_loss = train_loss / len(self.data)

        return train_loss

    def validation_pass(self):
        val_loss = 0

        self.net.eval()
        with torch.set_grad_enabled(False):
            for X, y in self.val_data:
                X = X.to(self.device)
                y = y.to(self.device)

                with autocast():
                    y_hat = self.net(X)
                    loss_value = self._loss_func(y_hat, y).item()

                val_loss += loss_value * len(y)  # scales to data size

        # scale to data size
        len_data = len(self.val_data)
        val_loss = val_loss / len_data

        return val_loss

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

class VdPTrainer(Trainer):
    """TODO.

    Attributes:
        net: the neural network to be trained.
        epochs: number of epochs to train the network.
        lr: learning rate.
        optimizer: optimizer (name of a optimizer inside `torch.optim`).
        loss_func: a valid PyTorch loss function.
        lr_scheduler: if a scheduler is to be used, provide the name of a valid
        `torch.optim.lr_scheduler`.
        lr_scheduler_params: parameters of selected `lr_scheduler`.
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
                 lr_scheduler_params: dict = None,
                 device=None, wandb_project="van-der-pol", wandb_group=None,
                 logger=None, checkpoint_every=50, random_seed=42) -> None:
        super().__init__(net, epochs, lr, optimizer, optimizer_params, loss_func,
                         lr_scheduler, lr_scheduler_params, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed)

        self.Nt = Nt
        self.Nf = Nf
        self.u_bounds = u_bounds
        self.y_bounds = y_bounds

        if h is None:
            self.h = lambda z: z * (self.y_bounds[1] - self.y_bounds[0]) / 2

        self.lamb = lamb

        # TODO: expose `val_params`
        y0 = torch.zeros(1,2).to(self.device)
        y0[0,1] = .1
        u = torch.zeros(1,1).to(self.device)
        dt = torch.ones(1,1).to(self.device) * 0.1
        K = 200
        self.val_params = {
            'y0': y0,
            'u': u,
            'dt': dt,
            'K': K,
        }

        self.data = None
        self.val_data = None

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
    def load_trainer(cls, net: nn.Module, run_id: str, wandb_project="van-der-pol", logger=None):
        self = super().load_trainer(net, run_id, wandb_project, logger)

        self._loss_func_y = eval(f"nn.{self.loss_func}()")
        self._loss_func_F = eval(f"nn.{self.loss_func}()")

        return self

    def setup_training(self):
        r =  super().setup_training()

        self._loss_func_y = eval(f"nn.{self.loss_func}()")
        self._loss_func_F = eval(f"nn.{self.loss_func}()")

        return r

    def prepare_data(self):
        (X_t, Y_t), (X_f, U_f) = get_data(self.Nt, self.Nf, self.u_bounds, self.y_bounds)

        X_t = X_t.to(self.device)
        Y_t = Y_t.to(self.device)
        X_f = X_f.to(self.device)
        U_f = U_f.to(self.device)

        self.data = (X_t, Y_t), (X_f, U_f)

        if self.val_data is None:
            (X_t, Y_t), (X_f, U_f) = get_data(self.Nt // 5, self.Nf // 5, self.u_bounds, self.y_bounds)

            X_t = X_t.to(self.device)
            Y_t = Y_t.to(self.device)
            X_f = X_f.to(self.device)
            U_f = U_f.to(self.device)

            dt, y0, u, K = self.val_params['dt'],self.val_params['y0'],self.val_params['u'],self.val_params['K']
            Y_ref = odeint(lambda t, y: f(y,u), y0, torch.Tensor([i * dt for i in range(K+1)]), method='rk4')

            self.val_data = (X_t, Y_t), (X_f, U_f), Y_ref

    def _run_epoch(self):
        # train
        train_time, (train_loss_y, train_loss_f) = timeit(self.train_pass)()
        train_loss = train_loss_y + self.lamb * train_loss_f

        self.l.info(f"Training pass took {train_time:.3f} seconds")
        self.l.info(f"Training loss = {train_loss}")

        # validation
        val_time, (iae, val_loss_y, val_loss_f) = timeit(self.validation_pass)()
        val_loss = val_loss_y + self.lamb * val_loss_f

        self.l.info(f"Validation pass took {val_time:.3f} seconds")
        self.l.info(f"IAE = {iae}")

        data_to_log = {
            "train_loss": train_loss,
            "train_loss_y": train_loss_y,
            "train_loss_f": train_loss_f,
            "val_loss": val_loss,
            "val_loss_y": val_loss_y,
            "val_loss_f": val_loss_f,
            "iae": iae,
        }

        val_score = iae  # defines best model

        return data_to_log, val_score

    def train_pass(self):
        self.net.train()

        (X_t, Y_t), (X_f, U_f) = self.data
        X_f.requires_grad_()
        with torch.set_grad_enabled(True):
            self._optim.zero_grad()

            with autocast():
                Y_t_pred = self.h(self.net(X_t))

                loss_y = self._loss_func_y(Y_t_pred, Y_t)

                Y_f_pred = self.h(self.net(X_f))

                dY_pred = self.get_dY_pred(Y_f_pred, X_f)

                dY_f = f(Y_f_pred, U_f)

                loss_f = self._loss_func_F(dY_pred, dY_f)

                loss = loss_y + self.lamb * loss_f

            self._scaler.scale(loss).backward()

            self._scaler.step(self._optim)
            self._scaler.update()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        return loss_y.item(), loss_f.item()

    def validation_pass(self):
        self.net.eval()

        (X_t, Y_t), (X_f, U_f), Y_ref = self.val_data

        dt, y0, u = self.val_params['dt'],self.val_params['y0'],self.val_params['u']
        Y = [self.val_params['y0'].cpu().detach().numpy().squeeze(),]

        x = torch.cat((dt,y0,u), dim=-1).to(self.device)
        with torch.set_grad_enabled(False):
            for _ in range(self.val_params['K']):
                y_next = self.h(self.net(x))

                Y.append(y_next.cpu().detach().numpy().squeeze())

                x = torch.cat((dt,y_next,u), dim=-1)

        iae = np.abs(Y_ref.cpu().detach().numpy() - Y).mean()

        X_f.requires_grad_()
        with torch.set_grad_enabled(False):
            Y_t_pred = self.h(self.net(X_t))

            loss_y = ((Y_t_pred - Y_t)**2).mean()

        with torch.set_grad_enabled(True):
            Y_f_pred = self.h(self.net(X_f))

            dY_pred = self.get_dY_pred(Y_f_pred, X_f)

        with torch.set_grad_enabled(False):
            dY_f = f(Y_f_pred, U_f)

            loss_f = ((dY_pred - dY_f)**2).mean()

        return iae, loss_y, loss_f

class VdPTrainerLBFGS(VdPTrainer):
    def __init__(self, net: nn.Module, Nt=1e3, Nf=1e5, u_bounds=(-1, 1),
                 y_bounds=(-3, 3), epochs=5, lr=1., lamb=.1, h=None,
                 lbfgs_params: dict = None, loss_func: str = 'MSELoss',
                 lr_scheduler: str = None, lr_scheduler_params: dict = None,
                 device=None, wandb_project="van-der-pol",
                 wandb_group=None, logger=None, checkpoint_every=50,
                 random_seed=42) -> None:
        super().__init__(net, Nt, Nf, u_bounds, y_bounds, epochs, lr, lamb, h, 'LBFGS',
                         lbfgs_params, loss_func, lr_scheduler, lr_scheduler_params,
                         device, wandb_project, wandb_group, logger,
                         checkpoint_every, random_seed)

    def train_pass(self):
        self.net.train()

        (X_t, Y_t), (X_f, U_f) = self.data
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

                dY_f = f(Y_f_pred, U_f)

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

class VdPTrainerPINN(Trainer):
    def __init__(self, net: nn.Module, y_0: np.ndarray, Nf=1e5, y_bounds=(-3, 3),
                 T_max=20., T_init=None, init_slack=2e3, end_slack=2e3, val_dt=0.1, epochs=5, lr=0.1, optimizer: str = 'Adam',
                 lamb=0.1, optimizer_params: dict = None, loss_func: str = 'MSELoss',
                 lr_scheduler: str = None, lr_scheduler_params: dict = None,
                 device=None, wandb_project="van-der-pol", wandb_group="PINN-Adam-ODE-like",
                 logger=None, checkpoint_every=50, random_seed=42):
        self.Nf = int(Nf)    # number of collocation points
        self.y_0 = y_0  # initial conditions
        self.y_bounds = y_bounds
        self.val_dt = val_dt

        self.T_max = T_max
        if T_init is None:
            self.T_init = 0.1 * self.T_max
        elif T_init < 0:
            self.T_init = self.T_max
        else:
            self.T_init = T_init
        
        if 0 < init_slack < 1:
            init_slack = init_slack * epochs

        if 0 < end_slack < 1:
            end_slack = end_slack * epochs

        self.init_slack = int(init_slack)
        self.end_slack = int(end_slack)

        if lamb is None:
            self.lamb = 1 / self.Nf
        else:
            self.lamb = lamb

        self.data = None
        self.val_data = None

        self._wandb_config = {
            'T_max': self.T_max,
            'T_init': self.T_init,
            'y0': self.y_0,
        }

        super().__init__(net, epochs, lr, optimizer, optimizer_params, loss_func,
                         lr_scheduler, lr_scheduler_params, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed)

    def prepare_data(self, T=None):
        if T is None:
            T = self.T_max

        X = torch.rand(self.Nf,1) * T

        self.data = X.to(self.device)

        if self.val_data is None:
            K = int(self.T_max / self.val_dt)

            X_val = torch.Tensor([i * self.val_dt for i in range(K+1)])

            u = torch.zeros(1, 1)  # uncontrolled
            Y_val = odeint(lambda t, y: f(y,u), torch.Tensor(self.y_0).unsqueeze(0), X_val, method='rk4')

            self.val_data = X_val.to(self.device).unsqueeze(-1), Y_val.to(self.device).squeeze()

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

    def _run_epoch(self):
        if self._e < self.init_slack:
            self.curr_T = self.T_init
        elif self._e > self.end_slack:
            self.curr_T = self.T_max
        else:
            self.curr_T = self.T_init + (self.T_max - self.T_init) * (self._e - self.init_slack) / (self.epochs - self.end_slack - self.init_slack)
        self.l.info(f"Current T = {self.curr_T:.2f}")

        self.prepare_data(T=self.curr_T)

        # train
        train_time, (train_loss_y, train_loss_f) = timeit(self.train_pass)()
        train_loss = train_loss_y + self.lamb * train_loss_f

        self.l.info(f"Training pass took {train_time:.3f} seconds")
        self.l.info(f"Training loss = {train_loss}")

        # validation
        val_time, (iae, partial_iae) = timeit(self.validation_pass)()

        self.l.info(f"Validation pass took {val_time:.3f} seconds")
        self.l.info(f"IAE = {iae}")
        self.l.info(f"Partial IAE = {partial_iae}")

        data_to_log = {
            "train_loss": train_loss,
            "train_loss_y": train_loss_y,
            "train_loss_f": train_loss_f,
            "iae": iae,
            "partial_iae": partial_iae,
            "T": self.curr_T,
        }

        val_score = iae  # defines best model

        return data_to_log, val_score

    def train_pass(self):
        self.net.train()

        # boundary
        X_t = torch.zeros(1,1).to(self.device)
        Y_t = torch.Tensor(self.y_0).unsqueeze(0).to(self.device)

        # collocation
        X_f = self.data

        X_t.requires_grad_()
        X_f.requires_grad_()
        with torch.set_grad_enabled(True):
            self._optim.zero_grad()

            with autocast():
                y_t_pred = self.net(X_t)

                dy_t_pred = torch.autograd.grad(
                    y_t_pred.sum(),
                    X_t,
                    create_graph=True,
                )[0]

                Y_t_pred = torch.stack([y_t_pred, dy_t_pred], dim=-1).squeeze(1)

                loss_y = self._loss_func(Y_t_pred, Y_t)

                y_pred = self.net(X_f)

                dy_pred = torch.autograd.grad(
                    y_pred.sum(),
                    X_f,
                    create_graph=True,
                )[0]

                ddy_pred = torch.autograd.grad(
                    dy_pred.sum(),
                    X_f,
                    create_graph=True,
                )[0]

                mu = 1.
                ddy = + mu * (1 - y_pred ** 2) * dy_pred - y_pred
                ode = ddy_pred - ddy
                loss_f = self._loss_func(ode, torch.zeros_like(ode))

                loss = loss_y + self.lamb * loss_f

            self._scaler.scale(loss).backward()

            self._scaler.step(self._optim)
            self._scaler.update()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        return loss_y.item(), loss_f.item()

    def validation_pass(self):
        self.net.eval()

        X, Y = self.val_data

        X.requires_grad_()
        with torch.set_grad_enabled(True):
            self._optim.zero_grad()
            y_pred = self.net(X)

        dy_pred = torch.autograd.grad(
            y_pred.sum(),
            X,
            create_graph=False,
        )[0]

        Y_pred = torch.stack([y_pred, dy_pred], dim=-1).squeeze(1)

        iae = (Y - Y_pred).abs().mean().item()

        partial_ix = (X <= self.curr_T).squeeze()
        X_part = X[partial_ix]
        Y_part = Y[partial_ix]

        X_part.requires_grad_()
        with torch.set_grad_enabled(True):
            self._optim.zero_grad()
            y_part_pred = self.net(X_part)

        dy_part_pred = torch.autograd.grad(
            y_part_pred.sum(),
            X_part,
            create_graph=False,
        )[0]

        Y_part_pred = torch.stack([y_part_pred, dy_part_pred], dim=-1).squeeze(1)

        partial_iae = (Y_part - Y_part_pred).abs().mean().item()

        if self._e % 500 == 0 and self._log_to_wandb:
            data = [[x,y1,y2] for x,y1,y2 in zip(
                X.squeeze().cpu().detach().numpy(),
                Y_pred[:,0].squeeze().cpu().detach().numpy(),
                Y_pred[:,1].squeeze().cpu().detach().numpy(),
            )]
            wandb.log({'dynamics': wandb.Table(data=data, columns=['t', 'y1', 'y2'])})

        return iae, partial_iae

class VdPTrainerPINNLBFGS(VdPTrainerPINN):
    def __init__(self, net: nn.Module, y_0: np.ndarray, Nf=100000, y_bounds=(-3, 3),
                 T_max=20, val_dt=0.1, epochs=5, lr=0.1, lamb=0.1,
                 optimizer_params: dict = None, loss_func: str = 'MSELoss',
                 lr_scheduler: str = None, lr_scheduler_params: dict = None,
                 device=None, wandb_project="van-der-pol", wandb_group="PINN-LBFGS",
                 logger=None, checkpoint_every=50, random_seed=42):
        super().__init__(net, y_0, Nf, y_bounds, T_max, val_dt, epochs, lr, 'LBFGS',
                         lamb, optimizer_params, loss_func, lr_scheduler,
                         lr_scheduler_params, device, wandb_project, wandb_group,
                         logger, checkpoint_every, random_seed)

    def train_pass(self):
        self.net.train()

        # boundary
        X_t = torch.zeros(1,1).to(self.device)
        Y_t = torch.Tensor(self.y_0).unsqueeze(0).to(self.device)

        # collocation
        X_f = self.data
        X_f.requires_grad_()

        y_losses = list()
        f_losses = list()
        losses = list()
        with torch.set_grad_enabled(True):
            def closure():
                self._optim.zero_grad()

                Y_t_pred = self.net(X_t)

                loss_y = self._loss_func(Y_t_pred, Y_t)

                Y_f_pred = self.net(X_f)

                dY_pred = self.get_dY_pred(Y_f_pred, X_f)

                u = torch.zeros(Y_f_pred.shape[0], 1).to(self.device)  # uncontrolled
                dY_f = f(Y_f_pred, u)

                loss_f = self._loss_func(dY_pred, dY_f)

                loss = loss_y + self.lamb * loss_f

                y_losses.append(loss_y.item())
                f_losses.append(loss_f.item())
                losses.append(loss.item())

                if loss.requires_grad:
                    loss.backward()

                return loss

            self._optim.step(closure)

            if self.lr_scheduler is not None:
                self._scheduler.step()

            y_losses = np.array(y_losses)
            f_losses = np.array(f_losses)
            # losses = np.array(losses)

        return y_losses[-1], f_losses[-1]

class VdPTrainerSIPINN(VdPTrainerPINN):
    """State-informed PINN.
    """
    def __init__(self, net: nn.Module, y_0: np.ndarray, Nf=100000, y_bounds=(-3, 3),
                 T_max=2., T_init=-1, init_slack=0, end_slack=0, val_dt=0.1,
                 epochs=5, lr=0.1, optimizer: str = 'Adam', lamb=0.1,
                 optimizer_params: dict = None, loss_func: str = 'MSELoss',
                 lr_scheduler: str = None, lr_scheduler_params: dict = None,
                 device=None, wandb_project="van-der-pol", wandb_group="PINN-SI",
                 logger=None, checkpoint_every=50, random_seed=42):
        super().__init__(net, y_0, Nf, y_bounds, T_max, T_init, init_slack, end_slack,
                         val_dt, epochs, lr, optimizer, lamb, optimizer_params,
                         loss_func, lr_scheduler, lr_scheduler_params, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed)

    def prepare_data(self, T=None):
        if T is None:
            T = self.T_max

        t = torch.rand(self.Nf,1) * T
        y0 = torch.Tensor(self.y_0).repeat(self.Nf,1)

        self.data = t.to(self.device), y0.to(self.device)

        if self.val_data is None:
            K = int(self.T_max / self.val_dt)

            t_val = torch.Tensor([i * self.val_dt for i in range(K+1)])

            y0_val = torch.Tensor(self.y_0).repeat(t_val.shape[0],1)

            u = torch.zeros(1, 1)  # uncontrolled
            Y_val = odeint(lambda t, y: f(y,u), torch.Tensor(self.y_0).unsqueeze(0), t_val, method='rk4')

            self.val_data = (t_val.to(self.device).unsqueeze(-1), y0_val.to(self.device)), Y_val.to(self.device).squeeze()

    def train_pass(self):
        self.net.train()

        # boundary
        t_b = torch.zeros(1,1).to(self.device)
        Y0_b = torch.Tensor(self.y_0).unsqueeze(0).to(self.device)

        # collocation
        t_f, y0_f = self.data

        t_b.requires_grad_()
        t_f.requires_grad_()
        with torch.set_grad_enabled(True):
            self._optim.zero_grad()

            with autocast():
                y_b_pred = self.net(t_b, Y0_b)

                dy_b_pred = torch.autograd.grad(
                    y_b_pred.sum(),
                    t_b,
                    create_graph=True,
                )[0]

                Y_b_pred = torch.cat([y_b_pred, dy_b_pred], dim=1)

                # as t=0, output must be equal to input
                loss_y = self._loss_func(Y_b_pred, Y0_b)

                y_pred = self.net(t_f, y0_f)

                dy_pred = torch.autograd.grad(
                    y_pred.sum(),
                    t_f,
                    create_graph=True,
                )[0]

                ddy_pred = torch.autograd.grad(
                    dy_pred.sum(),
                    t_f,
                    create_graph=True,
                )[0]

                mu = 1.
                ddy = + mu * (1 - y_pred ** 2) * dy_pred - y_pred
                ode = ddy_pred - ddy
                loss_f = self._loss_func(ode, torch.zeros_like(ode))

                loss = loss_y + self.lamb * loss_f

            self._scaler.scale(loss).backward()

            self._scaler.step(self._optim)
            self._scaler.update()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        return loss_y.item(), loss_f.item()

    def validation_pass(self):
        self.net.eval()

        (t, y0), Y = self.val_data

        t.requires_grad_()
        with torch.set_grad_enabled(True):
            self._optim.zero_grad()
            y_pred = self.net(t, y0)

        dy_pred = torch.autograd.grad(
            y_pred.sum(),
            t,
            create_graph=False,
        )[0]

        Y_pred = torch.stack([y_pred, dy_pred], dim=-1).squeeze(1)

        iae = (Y - Y_pred).abs().mean().item()

        partial_ix = (t <= self.curr_T).squeeze()
        t_part = t[partial_ix]
        y0_part = y0[partial_ix]
        Y_part = Y[partial_ix]

        t_part.requires_grad_()
        with torch.set_grad_enabled(True):
            self._optim.zero_grad()
            y_part_pred = self.net(t_part, y0_part)

        dy_part_pred = torch.autograd.grad(
            y_part_pred.sum(),
            t_part,
            create_graph=False,
        )[0]

        Y_part_pred = torch.stack([y_part_pred, dy_part_pred], dim=-1).squeeze(1)

        partial_iae = (Y_part - Y_part_pred).abs().mean().item()

        if self._e % 500 == 0 and self._log_to_wandb:
            data = [[x,y1,y2] for x,y1,y2 in zip(
                t.squeeze().cpu().detach().numpy(),
                Y_pred[:,0].squeeze().cpu().detach().numpy(),
                Y_pred[:,1].squeeze().cpu().detach().numpy(),
            )]
            wandb.log({'dynamics': wandb.Table(data=data, columns=['t', 'y1', 'y2'])})

        return iae, partial_iae

class VdPTrainerPINC(VdPTrainerSIPINN):
    def __init__(self, net: nn.Module, y_0: np.ndarray, Nf=100000, Nt=1000, y_bounds=(-3, 3),
                 T_max=20, T_init=-1, init_slack=0, end_slack=0, val_dt=0.1, epochs=5,
                 lr=0.1, optimizer: str = 'Adam', lamb=0.1, dt=1.,
                 optimizer_params: dict = None, loss_func: str = 'MSELoss',
                 lr_scheduler: str = None, lr_scheduler_params: dict = None,
                 device=None, wandb_project="van-der-pol", wandb_group="Uncontrolled-PINC",
                 logger=None, checkpoint_every=50, random_seed=42):
        super().__init__(net, y_0, Nf, y_bounds, T_max, T_init, init_slack, end_slack,
                         val_dt, epochs, lr, optimizer, lamb, optimizer_params,
                         loss_func, lr_scheduler, lr_scheduler_params, device,
                         wandb_project, wandb_group, logger, checkpoint_every,
                         random_seed)
        self.dt = dt
        self.Nt = int(Nt)

    def prepare_data(self, T=None):
        if T is None:
            T = self.T_max

        y_range = self.y_bounds[1] - self.y_bounds[0]

        t_b = torch.zeros(self.Nt,1).to(self.device)
        Y_b = torch.rand(self.Nt,2).to(self.device) * y_range + self.y_bounds[0]  # State data points for training

        t_f = torch.rand(self.Nf,1).to(self.device) * self.dt
        Y_f = torch.rand(self.Nf,2).to(self.device) * y_range + self.y_bounds[0]  # State data points for training

        self.data = (t_b, Y_b), (t_f, Y_f)

        if self.val_data is None:
            K = int(self.T_max / self.val_dt)

            t_val = torch.Tensor([i * self.val_dt for i in range(K+1)])

            u = torch.zeros(1, 1)  # uncontrolled
            Y_val = odeint(lambda t, y: f(y,u), torch.Tensor(self.y_0).unsqueeze(0), t_val, method='rk4')

            self.val_data = t_val.to(self.device).unsqueeze(-1), Y_val.to(self.device).squeeze()

    def train_pass(self):
        self.net.train()

        (t_b, Y0_b), (t_f, Y0_f) = self.data

        t_b.requires_grad_()
        t_f.requires_grad_()
        with torch.set_grad_enabled(True):
            self._optim.zero_grad()

            with autocast():
                y_b_pred = self.net(t_b, Y0_b)

                dy_b_pred = torch.autograd.grad(
                    y_b_pred.sum(),
                    t_b,
                    create_graph=True,
                )[0]

                Y_b_pred = torch.cat([y_b_pred, dy_b_pred], dim=1)

                loss_y = self._loss_func(Y_b_pred, Y0_b)  # as t=0, output must be equal to input

                y_pred = self.net(t_f, Y0_f)

                dy_pred = torch.autograd.grad(
                    y_pred.sum(),
                    t_f,
                    create_graph=True,
                )[0]

                ddy_pred = torch.autograd.grad(
                    dy_pred.sum(),
                    t_f,
                    create_graph=True,
                )[0]

                mu = 1.
                ddy = + mu * (1 - y_pred ** 2) * dy_pred - y_pred
                ode = ddy_pred - ddy
                loss_f = self._loss_func(ode, torch.zeros_like(ode))

                loss = loss_y + self.lamb * loss_f

            self._scaler.scale(loss).backward()

            self._scaler.step(self._optim)
            self._scaler.update()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        return loss_y.item(), loss_f.item()

    def validation_pass(self):
        self.net.eval()

        t_val, Y = self.val_data

        y_prev = torch.Tensor(self.y_0).unsqueeze(0).to(self.device)

        Y_pred = list()
        for _ in range(t_val.shape[0]):
            dt = torch.Tensor([self.val_dt]).unsqueeze(0).to(self.device)
            dt.requires_grad_()
            with torch.set_grad_enabled(True):
                self._optim.zero_grad()
                y_pred = self.net(dt, y_prev)

                dy_pred = torch.autograd.grad(
                    y_pred.sum(),
                    dt,
                    create_graph=False,
                )[0]

                y_prev = torch.stack([y_pred, dy_pred], dim=-1).squeeze(0)
                Y_pred.append(y_prev)

        Y_pred = torch.cat(Y_pred)

        iae = (Y - Y_pred).abs().mean().item()

        return iae, iae

class VdPTrainerPINCLBFGS(VdPTrainerPINC):
    def __init__(self, net: nn.Module, y_0: np.ndarray, Nf=100000, Nt=1000,
                 y_bounds=(-3, 3), T_max=20, T_init=-1, init_slack=0,
                 end_slack=0, val_dt=0.1, epochs=5, lr=0.1, lamb=0.1, dt=1.,
                 lbfgs_params: dict = None, loss_func: str = 'MSELoss',
                 lr_scheduler: str = None, lr_scheduler_params: dict = None,
                 device=None, wandb_project="van-der-pol",
                 wandb_group="Uncontrolled-PINC-LBFGS", logger=None,
                 checkpoint_every=50, random_seed=42):
        super().__init__(net, y_0, Nf, Nt, y_bounds, T_max, T_init, init_slack,
                         end_slack, val_dt, epochs, lr, 'LBFGS', lamb, dt,
                         lbfgs_params, loss_func, lr_scheduler,
                         lr_scheduler_params, device, wandb_project, wandb_group,
                         logger, checkpoint_every, random_seed)

    def train_pass(self):
        self.net.train()

        (t_b, Y0_b), (t_f, Y0_f) = self.data

        t_b.requires_grad_()
        t_f.requires_grad_()

        y_losses = list()
        f_losses = list()
        losses = list()
        with torch.set_grad_enabled(True):
            def closure():
                self._optim.zero_grad()

                y_b_pred = self.net(t_b, Y0_b)

                dy_b_pred = torch.autograd.grad(
                    y_b_pred.sum(),
                    t_b,
                    create_graph=True,
                )[0]

                Y_b_pred = torch.cat([y_b_pred, dy_b_pred], dim=1)

                loss_y = self._loss_func(Y_b_pred, Y0_b)  # as t=0, output must be equal to input

                y_pred = self.net(t_f, Y0_f)

                dy_pred = torch.autograd.grad(
                    y_pred.sum(),
                    t_f,
                    create_graph=True,
                )[0]

                ddy_pred = torch.autograd.grad(
                    dy_pred.sum(),
                    t_f,
                    create_graph=True,
                )[0]

                mu = 1.
                ddy = + mu * (1 - y_pred ** 2) * dy_pred - y_pred
                ode = ddy_pred - ddy
                loss_f = self._loss_func(ode, torch.zeros_like(ode))

                loss = loss_y + self.lamb * loss_f

                y_losses.append(loss_y.item())
                f_losses.append(loss_f.item())
                losses.append(loss.item())

                if loss.requires_grad:
                    loss.backward()

                return loss

            self._optim.step(closure)

            if self.lr_scheduler is not None:
                self._scheduler.step()

            y_losses = np.array(y_losses)
            f_losses = np.array(f_losses)
            # losses = np.array(losses)

        return y_losses[-1], f_losses[-1]
