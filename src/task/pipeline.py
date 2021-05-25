import pickle
from typing import Optional, Callable
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from ..data import CLMIDataset

class DataPipeline(LightningDataModule):
    def __init__(self, args) -> None:
        super(DataPipeline, self).__init__()
        self.args = args
        self.dataset_builder = CLMIDataset
        self.transform = transforms.Compose([         
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),                  
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                            )
                        ])
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                                                        self.dataset_builder,
                                                        self.args.feature_path,
                                                        "TRAIN",
                                                        self.transform
                                                        )

            self.val_dataset = DataPipeline.get_dataset(self.dataset_builder,
                                                        self.args.feature_path,
                                                        "VALID",
                                                        self.transform
                                                        )

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(self.dataset_builder,
                                                        self.args.feature_path,
                                                        "TEST",
                                                        self.transform
                                                        )

    def train_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(self.train_dataset,
                                           batch_size=self.args.batch_size,
                                           num_workers=self.args.num_workers,
                                           drop_last=True,
                                           shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(self.val_dataset,
                                           batch_size=self.args.batch_size,
                                           num_workers=self.args.num_workers,
                                           drop_last=True,
                                           shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(self.test_dataset,
                                           batch_size=1,
                                           num_workers=self.args.num_workers,
                                           drop_last=False,
                                           shuffle=False)

    @classmethod
    def get_dataset(cls, dataset_builder:Callable, feature_path, split, transform) -> Dataset:
        dataset = dataset_builder(feature_path, split, transform)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset ,batch_size: int, num_workers: int, shuffle: bool, drop_last: bool,
                       **kwargs) -> DataLoader:
        return DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          **kwargs)