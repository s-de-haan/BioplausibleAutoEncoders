import datetime
from typing import Any, Dict
import torch
import os
import logging

from copy import deepcopy
from torch.utils.data import DataLoader
from callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)
from utils import dotdict


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train, eval, config, callbacks=None) -> None:
        self.model = model
        self.callbacks = callbacks
        self.config = config
        self.device = self.config.device

        if config.device == "cuda":
            self.model = torch.nn.DataParallel(self.model)
        model.to(self.device)

        self.train_loader = DataLoader(
            dataset=train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

        self.eval_loader = DataLoader(
            dataset=eval,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

    def train(self) -> None:
        self._prepare_training()
        self.callback_handler.on_train_begin(training_config=self.config)

        logger.info(
            msg=f"Training:\n - epochs: {self.config.epochs}\n - batch_size: {self.config.batch_size}\n"
        )

        # TODO log to output dir with get_file_logger
        best_train_loss = 1e10
        best_eval_loss = 1e10

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
            elif epoch_train_loss < best_train_loss and self.eval_loader is None:
                best_train_loss = epoch_train_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            self.callback_handler.on_epoch_end(training_config=self.config)
            self.callback_handler.on_log(
                self.config,
                metrics,
                logger=logger,
                epoch=epoch,
            )

        final_dir = os.path.join(self.training_dir, "final_model")
        self._save_model(best_model, dir_path=final_dir)

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
            inputs = self._set_inputs_to_device(inputs)

            model_output = self.model(inputs)

            loss = model_output.loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(training_config=self.config)

        # self.model.module.update()

        epoch_loss /= len(self.train_loader)

        return epoch_loss

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
            inputs = self._set_inputs_to_device(inputs)

            with torch.no_grad():
                model_output = self.model(inputs)

            loss = model_output.loss

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in eval loss")

            self.callback_handler.on_eval_step_end(training_config=self.config)

        epoch_loss /= len(self.eval_loader)

        return epoch_loss

    def _set_inputs_to_device(self, inputs: Dict[str, Any]):
        inputs_on_device = inputs

        # if self.device == "cuda":
        #     cuda_inputs = dict.fromkeys(inputs)

        #     for key in inputs.keys():
        #         if torch.is_tensor(inputs[key]):
        #             cuda_inputs[key] = inputs[key].cuda()

        #         else:
        #             cuda_inputs[key] = inputs[key]
        #     inputs_on_device = cuda_inputs

        return inputs_on_device

    def _save_model(self, model, dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model.save(dir_path)

        self.training_config.save_json(dir_path, "training_config")

        self.callback_handler.on_save(self.training_config)

    def _get_file_logger(self, output_dir: str):
        if not os.path.exists(output_dir) and self.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        log_name = f"training_logs_{self._training_signature}"

        file_logger = logging.getLogger(log_name)
        file_logger.setLevel(logging.INFO)
        f_handler = logging.FileHandler(
            os.path.join(output_dir, f"training_logs_{self._training_signature}.log")
        )
        f_handler.setLevel(logging.INFO)
        file_logger.addHandler(f_handler)

        # Do not output logs in the console
        file_logger.propagate = False

        return file_logger

    def _prepare_training(self):
        self._set_seed(self.config.seed)
        self._set_optimizer()
        self._set_output_dir()
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

    def _set_output_dir(self):
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self._training_signature = (
            str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        )

        training_dir = os.path.join(
            self.config.output_dir,
            f"{self.model.module.name}_training_{self._training_signature}",
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
