from typing import Dict, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm

from src.common.utils import PROJECT_ROOT


class MyDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, train: bool, **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.train = train

        image_tranforms = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
        self.cifar10 = CIFAR10(
            root=path, train=train, transform=image_tranforms, download=True
        )

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.cifar10[index]

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: MyDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.test[0], _recursive_=False
    )
    print(cfg)


if __name__ == "__main__":
    main()
