import torch
import polars as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule



# =====================================================
#
#                 DATASET DEFINITION
#
# =====================================================

class CustomDataset(Dataset):
    def __init__(self,
                 path: Path,
                 columns: list[str],
                 target: str):
        self.path = path
        self.columns = columns
        self.target = target
        self.dataframe = pl.scan_parquet(path).select(self.columns + [self.target])


    def __len__(self) -> int:
        """Returns the length of the Dataset in a lazy fashion.

        Returns:
            - int : Length of the Dataset.
        """
        return self.dataframe.count().select(self.target).collect().item()


    def __getitem__(self,
                    idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns in a torch.Tensor format the inputs and the target from a specific row with id equal to idx, all in a lazy fashion.

        Args:
            - idx (int): The index of the wanted row.

        Returns:
            - tuple[torch.Tensor, torch.Tensor] : The inputs and target as a tuple of tensors.
        """
        # Get row by index in a lazy fashion
        row = self.dataframe.slice(idx, 1).collect().row(0, named=True) 

        # Get the input
        inputs = [elem for col in self.columns for elem in row[col]]

        # Get the target
        target = row[self.target]

        return torch.tensor(inputs, dtype=torch.float), torch.tensor(target, dtype=torch.float)



# =====================================================
#
#                DATAMODULE DEFINITION
#
# =====================================================
    
class DataModule(LightningDataModule):
    def __init__(self,
                 path : str,
                 batch_size: int = 128) -> None:
        super().__init__()

        self.path = Path(path)
        self.batch_size = batch_size
        pass


    def prepare_data(self):
        """
        """
        pass


    def setup(self, stage=None):
        """
        """
        if stage == 'fit' or stage is None:
            ...
        if stage == 'test' or stage is None:
            ...
        pass


    def train_dataloader(self) -> DataLoader:
        """
        """
        pass


    def test_dataloader(self) -> DataLoader:
        """
        """
        pass


    def val_dataloader(self) -> DataLoader:
        """
        """
        pass



if __name__ == "__main__":
    pass
