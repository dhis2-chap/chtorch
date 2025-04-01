from typing import Any

import lightning as L
import torch
from torch import optim


class DeepARLightningModule(L.LightningModule):
    def __init__(self, module, loss, weight_decay=1e-6):
        super().__init__()
        self.module = module
        self.loss = loss
        self.weight_decay = weight_decay
        self.last_validation_loss = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        X, locations, y, population = batch
        log_rate = self.module(X, locations).squeeze(-1)
        loss = self.loss(log_rate, y, population)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, locations, y, population = batch
        log_rate = self.module(X, locations).squeeze(-1)
        loss = self.loss(log_rate, y, population)
        self.last_validation_loss = loss
        self.log("validation_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        return optim.AdamW([
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}
        ], lr=1e-3)
        #
        #
        # return optim.AdamW(self.parameters(),
        #                    lr=1e-3,
        #                    weight_decay=self.weight_decay)
