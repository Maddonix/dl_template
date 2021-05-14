from typing import Any, List

import torch
import torch.nn as nn
import torchvision
from pytorch_lightning import LightningModule
from src.models.modules.audio_preprocessing import AudioPreprocess
from src.utils.metric_utils import get_f1, metric_to_dict


class AudioLitModel(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        classes: List,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        data_dir: str = "data/",
        sr: int = 41000,
        duration: int = 10000,  # in ms
        n_mels: int = 64,
        n_fft: int = 1024,
        top_db: int = 80,
        n_mfcc: int = 64,
        hop_len: int = 512,
        **kwargs
    ):
        super().__init__()
        self.data_path = data_dir
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.top_db = top_db
        self.n_mfcc = n_mfcc
        self.hop_len = hop_len
        self.classes = classes
        self.num_classes = len(self.classes)

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # self.model = SimpleConvNet(hparams=self.hparams)
        self.preprocess = AudioPreprocess(hparams=self.hparams)
        self.model = torchvision.models.resnext50_32x4d()
        # Change first layer
        self.model.conv1 = nn.Conv2d(
            1,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False,
        )
        # Change last layer
        self.model.fc = nn.Linear(
            self.model.fc.in_features, self.num_classes, bias=True
        )

        self.model = nn.Sequential(self.preprocess, self.model)

        # loss function
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss_hist = {
            "train": [],
            "val": [],
            "test":[]
        }
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.f1_train, self.f1_val, self.f1_test = get_f1(self.num_classes)
        self.f1_hist = {
            "train": [],
            "val": [],
            "test":[]
        }


    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.type_as(logits))
        preds = logits.clone().detach()
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log train metrics
        f1 = self.f1_train(preds, targets)
        f1 = metric_to_dict(self.classes, f1)
        self.log("train/f1", f1, on_step = False, on_epoch = True, prog_bar = False)
        self.log("train/loss", loss, on_step = False, on_epoch = True, prog_bar = False)

        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.f1_hist["train"].append(self.trainer.callback_metrics["train/f1"])
        self.loss_hist["train"].append(self.trainer.callback_metrics["train/loss"])

        self.log("train/loss_best", min(self.loss_hist["train"]), prog_bar = False)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        f1 = self.f1_val(preds, targets)
        f1 = metric_to_dict(self.classes, f1)
        self.log("val/f1", f1, on_step = False, on_epoch = True, prog_bar = False)
        self.log("val/loss", loss, on_step = False, on_epoch = True, prog_bar = False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        self.f1_hist["val"].append(self.trainer.callback_metrics["val/f1"])
        self.loss_hist["val"].append(self.trainer.callback_metrics["val/loss"])

        self.log("val/loss_best", min(self.loss_hist["val"]), prog_bar = False)


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        f1 = self.f1_test(preds, targets)
        f1 = metric_to_dict(self.classes, f1)
        self.log("test/f1", f1, on_step = False, on_epoch = True, prog_bar = False)
        self.log("test/loss", loss, on_step = False, on_epoch = True, prog_bar = False)
        # log test metrics
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss_best",
        }
