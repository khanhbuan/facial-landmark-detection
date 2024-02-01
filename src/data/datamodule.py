import torch
from torch.utils.data import Dataset
from lightning import LightningDataModule
import albumentations as A
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split
import hydra
import rootutils
from omegaconf import DictConfig
from src.data.components.customed_dataset import Customed_Dataset
from src.data.components.transformed_dataset import Transformed_Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class DataModule(LightningDataModule):
    def __init__(
        self,
        train_val_split: Tuple[int, int] = (5666, 1000),
        batch_size: int = 64,
        num_workers: int = 3,
        train_transform: Optional[A.Compose] = None,
        val_test_transform: Optional[A.Compose] = None,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        self.train_transform = train_transform

        self.val_test_transform = val_test_transform

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    # stage: Optional[str] = None
    def setup(self):
        train_val_dataset = Customed_Dataset(is_train=True)
        test_dataset = Customed_Dataset(is_train=False)
        train_dataset, val_dataset = random_split(
            dataset = train_val_dataset,
            lengths = self.hparams.train_val_split,
            generator = torch.Generator().manual_seed(42),
        )
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = Transformed_Dataset(train_dataset, transform=self.train_transform)
            self.data_val = Transformed_Dataset(val_dataset, transform=self.val_test_transform)
            self.data_test = Transformed_Dataset(test_dataset, transform=self.val_test_transform)
    
    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="300w")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule: LightningDataModule = hydra.utils.instantiate(config=cfg)
    datamodule.setup
    print(datamodule.train_transform)
    """
    for batch in datamodule.train_dataloader():
        images, labels = batch
        print(labels)
    """

if __name__ == "__main__":
    main()