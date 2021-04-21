import itertools
from typing import Any, Dict, Sequence, Tuple, Union
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning.metrics import Accuracy
from torch.optim import Optimizer

from src.common.utils import PROJECT_ROOT
from src.common.utils import iterate_elements_in_batches
from src.common.utils import render_images
from src.pl_modules.simple_cnn import CNN
from torch.nn import functional as F


class MyModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        self.cnn = CNN(
            input_channels=self.hparams.input_channels, n_feature=self.hparams.n_feature
        )

        metric = Accuracy()
        self.train_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

    def forward(self, images: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        return self.cnn(images)

    def step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int):
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return {"logits": logits, "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        out_step = self.step(x, y, batch_idx)
        self.train_accuracy(torch.softmax(out_step["logits"], dim=-1), y)
        self.log_dict(
            {"train_loss": out_step["loss"], "train_acc": self.train_accuracy}
        )
        return out_step["loss"]

    # def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
    #     loss = self.step(batch, batch_idx)
    #     self.log_dict(
    #         {"val_loss": loss},
    #         on_step=False,
    #         on_epoch=True,
    #         prog_bar=True,
    #     )
    #     return loss

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        out_step = self.step(x, y, batch_idx)
        self.test_accuracy(torch.softmax(out_step["logits"], dim=-1), y)
        self.log_dict({"test_loss": out_step["loss"], "test_acc": self.test_accuracy})
        return {"image": x, "y_true": y, **out_step}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        images = []

        for output_element in itertools.islice(
            iterate_elements_in_batches(
                outputs=outputs, elements_to_unbatch=["image", "logits", "y_true"]
            ),
            20,
        ):
            rendered_images = render_images(output_element["image"], autoshow=False)
            caption = f'y_pred: {output_element["logits"].argmax(-1)} [gt: {output_element["y_true"]}]'
            images.append(wandb.Image(rendered_images, caption=caption))
        self.logger.experiment.log({"Test imgaes:": images})

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
