import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import AUROC

from src.utils.torch_utils import collate


class DefaultLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        compile_model: bool,
        scheduler_interval: str = "step",
    ) -> None:
        super().__init__()

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.compile_model = compile_model
        self.scheduler_interval = scheduler_interval  # step or epoch

        # metric objects for calculating and averaging auroc across batches
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_auroc_best = MaxMetric()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, batch: dict) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_auroc.reset()
        self.val_auroc_best.reset()

    def model_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        batch = collate(batch)
        logits, y = self.forward(batch=batch), batch["labels"]
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_auroc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_auroc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        auroc_score = self.val_auroc.compute()  # type: ignore
        self.val_auroc_best(auroc_score)  # update best
        # log `val_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/auroc_best", self.val_auroc_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile_model and stage == "fit":
            self.net = torch.compile(model=self.net)  # type: ignore

    def configure_optimizers(self) -> dict:  # type: ignore
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())  # type: ignore
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer, num_training_steps=self.trainer.estimated_stepping_batches)  # type: ignore
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": self.scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class EmbeddingLitModule(LightningModule):
    def __init__(self, net: torch.nn.Module, compile_model: bool) -> None:
        super().__init__()

        self.net = net
        self.compile_model = compile_model

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net(batch)

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == "fit":
            self.net = torch.compile(model=self.net)  # type: ignore
