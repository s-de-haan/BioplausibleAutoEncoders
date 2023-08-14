import torch
import os

from torch.utils.data import DataLoader
from typing import Any
from callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)

class TrainingPipeline():

    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config

        self.model.to(config["device"])

        if config["device"] == "cuda":
            self.model = torch.nn.DataParallel(self.model)

    def __call__(self, train, eval = None, callbacks= None) -> None:

        # Possibly create datasets from train and eval
        trainer = self.Trainer(
            model=self.model,
            train_dataset=train,
            eval_dataset=eval,
            callbacks=callbacks,
        )

        trainer.train()

    class Trainer():

        def __init__(self, model, train, eval, callbacks) -> None:
            self.model = model
            self.callbacks = callbacks
            self.device = self.config["device"]

            model.to(self.device)

            train_loader = DataLoader(
                dataset=train,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
            )

            eval_loader = DataLoader(
                dataset=eval,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
            )

        def prepare_training(self):
            self.set_seed(self.config.seed)
            self.set_optimizer()
            self._set_output_dir()
            self._setup_callbacks()

        def set_seed(seed: int):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        def set_optimizer(self):
            if self.config["optimizer"] == "Adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
            elif self.config["optimizer"] == "SGD":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["lr"])
            else:
                raise NotImplementedError
            
        def _set_output_dir(self):
            self.output_dir = self.config["output_dir"]
            os.makedirs(self.output_dir, exist_ok=True)

        def _setup_callbacks(self):
            if self.callbacks is None:
                self.callbacks = [TrainingCallback()]

            self.callback_handler = CallbackHandler(
                callbacks=self.callbacks, model=self.model
            )

            self.callback_handler.add_callback(ProgressBarCallback())
            self.callback_handler.add_callback(MetricConsolePrinterCallback())
        
