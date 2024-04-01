import polars
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader



# =====================================================
#
#                 DATASET DEFINITION
#
# =====================================================

class CustomDataset(Dataset):
    def __init__(self,
                 path: Path):
        self.path = path
        self.dataframe = polars.read_csv(path)


    def __len__(self) -> int:
        """Returns the length of the Dataset.

        Returns:
            - int : Length of the Dataset.
        """"
        ...


    def __getitem__(self,
                    idx: int):
        ...



# =====================================================
#
#                DATAMODULE DEFINITION
#
# =====================================================
    
class DataModule(pl.LightningDataModule):
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
