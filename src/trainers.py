import datetime
import json
import torch
import os
import logging

from copy import deepcopy
from torch.utils.data import DataLoader
from src.callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)
from src.utils import dotdict

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train, eval, config, callbacks=None) -> None:
        self.model = model
        self.callbacks = callbacks
        self.config = config
        self.device = self.config.device

        self._set_device(self.config.device)

        model.to(self.device)

        self.train_loader = DataLoader(
            dataset=train,
            batch_size=self.config.batch_size,
            generator=torch.Generator(device=self.device).manual_seed(self.config.seed),
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        self.eval_loader = DataLoader(
            dataset=eval,
            batch_size=self.config.batch_size,
            generator=torch.Generator(device=self.device).manual_seed(self.config.seed),
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def train(self) -> None:
        self._prepare_training()
        self.callback_handler.on_train_begin(training_config=self.config)

        logger.info(
            msg=f"Training:\n - epochs: {self.config.epochs}\n - batch_size: {self.config.batch_size}\n - optimizer: {self.config.optimizer}\n - scheduler: {self.config.scheduler}\n - device: {self.config.device}\n - output_dir: {self.config.output_dir}\n - seed: {self.config.seed}\n - encoder_layers: {self.config.encoder_layers}\n - decoder_layers: {self.config.decoder_layers}\n - learning_rate: {self.config.lr}\n - gamma: {self.config.gamma}\n - patience: {self.config.patience}\n - num_workers: {self.config.num_workers}\n - training_dir: {self.training_dir}\n - model: {self.model.name}\n"
        )

        # TODO log to output dir with get_file_logger
        best_train_loss = 1e10
        best_eval_loss = 1e10

        self.model.zero_grad()

        for epoch in range(1, self.config.epochs + 1):
            self.callback_handler.on_epoch_begin(
                training_config=self.config,
                epoch=epoch,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
            )

            metrics = dotdict()

            epoch_train_loss = self._train_step(epoch)
            metrics.epoch_train_loss = epoch_train_loss

            if self.eval_loader is not None:
                epoch_eval_loss = self._eval_step(epoch)
                metrics.epoch_eval_loss = epoch_eval_loss

            if epoch_eval_loss < best_eval_loss:
                best_eval_loss = epoch_eval_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model
            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss

            self.callback_handler.on_epoch_end(training_config=self.config)
            self.callback_handler.on_log(
                self.config,
                metrics,
                logger=logger,
                epoch=epoch,
            )

        self._save_model(best_model, dir_path=self.training_dir)
        logger.info(f"\nBest train loss: {best_train_loss}, Best eval loss: {best_eval_loss}")

    def _train_step(self, epoch: int):
        """The trainer performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """
        self.callback_handler.on_train_step_begin(
            training_config=self.config,
            train_loader=self.train_loader,
            epoch=epoch,
        )

        self.model.train()

        epoch_loss = 0

        for inputs in self.train_loader:
            inputs = inputs.to(self.device)
            model_output = self.model(inputs)

            loss = model_output.loss

            self.optimizer.zero_grad()
            self.model.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(training_config=self.config)

        epoch_loss /= len(self.train_loader)

        return epoch_loss

    @torch.no_grad()
    def _eval_step(self, epoch: int):
        """Perform an evaluation step

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The evaluation loss
        """

        self.callback_handler.on_eval_step_begin(
            training_config=self.config,
            eval_loader=self.eval_loader,
            epoch=epoch,
        )

        self.model.eval()

        epoch_loss = 0

        for inputs in self.eval_loader:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                model_output = self.model(inputs)

            loss = model_output.loss

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in eval loss")

            self.callback_handler.on_eval_step_end(training_config=self.config)

        epoch_loss /= len(self.eval_loader)

        return epoch_loss


    def _set_device(self, device: str):
        self.device = torch.device(device)
        torch.set_default_device(self.device)

    def _save_model(self, model, dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        torch.save(model.state_dict(), os.path.join(dir_path, "model.pt"))

        with open(os.path.join(dir_path, "config.json"), "w") as fp:
            json.dump(self.config, fp)

        self.callback_handler.on_save(self.config)

    def _setup_logger(self):
        logging.basicConfig(filename=os.path.join(self.training_dir, "training.log"), level=logging.INFO, format='%(message)s')

    def _prepare_training(self):
        self._set_seed(self.config.seed)
        self._set_optimizer()
        self._set_scheduler()
        self._set_output_dir()
        self._setup_logger()
        self._setup_callbacks()

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _set_optimizer(self):
        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config["lr"]
            )
        elif self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config["lr"]
            )
        else:
            raise NotImplementedError

    def _set_scheduler(self):
        if self.config.scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.lr, gamma=self.config.gamma
            )
        elif self.config.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.gamma,
                patience=self.config.patience,
                verbose=True,
            )
        elif self.config.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler == None:
            pass
        else:
            raise NotImplementedError

    def _set_output_dir(self):
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self._training_signature = (
            str(datetime.datetime.now())[5:19].replace(" ", "_").replace(":", "-")
        )

        training_dir = os.path.join(
            self.config.output_dir,
            f"{self.model.name}_lr{self.config.lr}_{self._training_signature}",
        )

        self.training_dir = training_dir

        if not os.path.exists(training_dir):
            os.makedirs(training_dir, exist_ok=True)

    def _setup_callbacks(self):
        if self.callbacks is None:
            self.callbacks = [TrainingCallback()]

        self.callback_handler = CallbackHandler(
            callbacks=self.callbacks, model=self.model
        )

        self.callback_handler.add_callback(ProgressBarCallback())
        self.callback_handler.add_callback(MetricConsolePrinterCallback())
