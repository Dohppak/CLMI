import torch
import torch.nn as nn

from omegaconf import DictConfig
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pytorch_lightning import LightningModule
from .utils import CosineAnnealingWarmupRestarts

class ContrastiveRunner(LightningModule):
    def __init__(self, model: nn.Module, args):
        super().__init__()
        self.model = model
        self.args = args
        self.image_criterion = nn.CrossEntropyLoss()
        self.audio_criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate,
            weight_decay= self.args.weight_decay,
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=opt,
            first_cycle_steps=10000,
            cycle_mult=1.0,
            max_lr=self.args.learning_rate,
            min_lr=1e-6,
            warmup_steps=2000,
            gamma=1.0
        )
        
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


    def training_step(self, batch, batch_idx):
        audio, image = batch
        device = audio.device
        logits_per_audio, logits_per_image = self.model(audio, image)
        ground_truth = torch.arange(audio.shape[0]).long().to(device)
        loss_audio= self.audio_criterion(logits_per_audio, ground_truth)
        loss_image= self.image_criterion(logits_per_image, ground_truth)
        loss = (loss_audio + loss_image)/2
        self.log_dict({'loss': loss}, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, image = batch
        device = audio.device
        logits_per_audio, logits_per_image = self.model(audio, image)
        ground_truth = torch.arange(audio.shape[0]).long().to(device)
        loss_audio= self.audio_criterion(logits_per_audio, ground_truth)
        loss_image= self.image_criterion(logits_per_image, ground_truth)
        loss = (loss_audio + loss_image)/2
        outputs = {'val_loss': loss, 'loss_audio': loss_audio, 'loss_image': loss_image}
        return outputs
    
    def validation_step_end(self, batch_parts):
        return batch_parts

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        loss_audio = torch.mean(torch.stack([output["loss_audio"] for output in outputs]))
        loss_image = torch.mean(torch.stack([output["loss_image"] for output in outputs]))
        outputs = {'val_loss': val_loss, 'loss_audio': loss_audio, 'loss_image': loss_image}
        self.log_dict(outputs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return outputs