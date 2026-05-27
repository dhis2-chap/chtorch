from typing import Any

import lightning as L
import torch
from torch import optim
import logging

from chtorch.configuration import ModelConfiguration

logger = logging.getLogger(__name__)


class DeepARLightningModule(L.LightningModule):
    def __init__(self, module, loss, target_scaler=None, cfg: ModelConfiguration=None):
        super().__init__()
        self.module = module
        self.loss = loss
        self.weight_decay = cfg.weight_decay
        self.learning_rate = cfg.learning_rate
        self.last_validation_losses = {}
        self.last_train_losses = {}
        self._target_scaler = target_scaler
        self.save_hyperparameters(cfg.model_dump())
        self.cfg = cfg

    @property
    def last_validation_loss(self):
        if not self.last_validation_losses:
            return float('nan')
        return float(sum(self.last_validation_losses.values()) / len(self.last_validation_losses))

    @property
    def last_train_loss(self):
        if not self.last_train_losses:
            return float('nan')
        return float(sum(self.last_train_losses.values()) / len(self.last_train_losses))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        eta, past_eta = self.module(batch.X, batch.locations)
        if self._target_scaler is not None:
            log_rate = self._target_scaler.scale_by_location(batch.locations[:, 0, 0], eta)
            past_log_rate = self._target_scaler.scale_by_location(batch.locations[:, 0, 0], past_eta)
        else:
            log_rate = eta
            past_log_rate = past_eta
        loss = self.loss(log_rate, batch.y, batch.population) + self.cfg.past_ratio * self.loss(past_log_rate,
                                                                                batch.past_y[:, 1:],
                                                                                   batch.population)
        self.last_train_losses[batch_idx] = loss.detach().item()
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            log_rate, *_ = self.module(batch.X, batch.locations)
            assert not torch.isnan(log_rate).any()
            if self._target_scaler is not None:
                log_rate = self._target_scaler.scale_by_location(batch.locations[:, 0, 0], log_rate)

            loss = self.loss(log_rate, batch.y, batch.population)
            self.last_validation_losses[batch_idx] = loss.detach().item()
            self.log("validation_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            return loss

    def configure_optimizers(self):
        decay_dict = self._get_decay_dict()
        optimizer = optim.AdamW(decay_dict, lr=self.learning_rate)
        logger.info('Using learning rate %s', self.learning_rate)
        return optimizer

    def _get_decay_dict(self):
        # Three buckets:
        #   no_decay: biases & norm layers
        #   embed:    every embedding gets 10× weight_decay
        #   decay:    everything else
        # The previous implementation also applied 100× to params whose name
        # contained '.0.', which matched the primary location embedding
        # (`location_embeddings.0.weight`) — the very thing we want the model
        # to learn most aggressively — and only 10× to the smaller parent
        # embedding at index 1. That was almost certainly the inverse of the
        # author's intent; collapsing to a single 'embed' bucket avoids the
        # foot-gun.
        decay = []
        embed_decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "norm" in name.lower():
                no_decay.append(param)
            elif 'embed' in name:
                embed_decay.append(param)
            else:
                decay.append(param)
        decay_dict = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": embed_decay, "weight_decay": self.weight_decay * 10},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        logger.info(f"Decay params: {len(decay)}, embed: {len(embed_decay)}, no decay: {len(no_decay)}")
        return decay_dict
