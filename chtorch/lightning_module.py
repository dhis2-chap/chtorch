from typing import Any

import lightning as L
import torch
from torch import optim
import logging

from chtorch.count_transforms import IdentityTransform
from chtorch.distribution_loss import PoissonLoss

logger = logging.getLogger(__name__)


class DeepARLightningModule(L.LightningModule):
    def __init__(self, module, loss, target_scaler=None, weight_decay=1e-6, learning_rate=1e-3):
        super().__init__()
        self.module = module
        self.loss = loss
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.last_validation_losses = {}
        self.last_train_losses = {}
        self._target_scaler = target_scaler

    @property
    def last_validation_loss(self):
        return sum(self.last_validation_losses.values()) / len(
            self.last_validation_losses) if self.last_validation_losses else 0

    @property
    def last_train_loss(self):
        return sum(self.last_train_losses.values()) / len(self.last_train_losses)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        #X, locations, y, population = batch
        eta = self.module(batch.X, batch.locations).squeeze(-1)
        if self._target_scaler is not None:
            log_rate = self._target_scaler.scale_by_location(batch.locations[:, 0, 0], eta)
        else:
            log_rate = eta
        loss = self.loss(log_rate, batch.y, batch.population)
        self.last_train_losses[batch_idx] = loss
        if batch_idx == 0:
            self.log("train_loss", self.last_train_loss, prog_bar=True, logger=True)
        return loss

    def _debug(self, batch_idx, loss, y):
        with torch.no_grad():
            y_tmp = y.detach()
            min_loss = PoissonLoss(IdentityTransform()).forward(y_tmp[..., None], y_tmp, y_tmp).mean()
            # print(min_loss.mean())
            print(f"Batch {batch_idx} loss: {loss.item()}, min_loss: {min_loss.mean()} n values: {len(y_tmp)}")

    def validation_step(self, batch, batch_idx):
        #X, locations, y, population = batch
        log_rate = self.module(batch.X, batch.locations).squeeze(-1)
        if self._target_scaler is not None:
            log_rate = self._target_scaler.scale_by_location(batch.locations[:, 0, 0], log_rate)
        loss = self.loss(log_rate, batch.y, batch.population)
        self.last_validation_losses[batch_idx] = loss
        self.log("validation_loss", loss, prog_bar=True, logger=True, on_step=False)
        return loss

    def configure_optimizers(self):
        decay_dict = self._get_decay_dict()
        optimizer = optim.AdamW(decay_dict, lr=self.learning_rate)
        logger.info('Using learning rate %s', self.learning_rate)
        return optimizer

    def _get_decay_dict(self):
        decay = []
        embed_decay = []
        level_2_decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "norm" in name.lower():
                no_decay.append(param)
            elif 'embed' in name:
                if '.0.' in name:
                    level_2_decay.append(param)
                else:
                    embed_decay.append(param)
            else:
                decay.append(param)
        decay_dict = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": embed_decay, "weight_decay": self.weight_decay * 10},
            {"params": level_2_decay, "weight_decay": self.weight_decay * 100},
            {"params": no_decay, "weight_decay": 0.0}
        ]
        return decay_dict
