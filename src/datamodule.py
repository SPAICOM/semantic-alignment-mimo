"""In this python module we define class that handles the dataset:
    - CustomDataset: a custom Pytorch Dataset.
    - DataModule: a Pytorch Lightning Data Module.
"""

import torch
import polars as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule



# =====================================================
#
#                 DATASET DEFINITION
#
# =====================================================

class CustomDataset(Dataset):
    """A custom implementation of a Pytorch Dataset.

    Args:
        - path (Path): The path towards the parquet file containg the data.
        - colums (list[str]): The list containg the columns name of the input data.
                            The input will be concatenated.
        - target (str): The name of the target column.

    Attributes:
        - self.path (Path): Where the path argument is stored.
        - self.columns (list[str]): Where the columns argument is stored.
        - self.target (str): Where the target argument is stored.
        - self.datataframe (pl.LazyFrame): The scan of the parquet file.
    """
    def __init__(self,
                 path: Path,
                 columns: list[str],
                 target: str):
        self.path = path
        self.columns = columns
        self.target = target
        self.dataframe = pl.scan_parquet(path).select(self.columns + [self.target])

        # Get the input and the output size
        row = self.dataframe.slice(0, 1).collect().row(0, named=True) 
        input = [elem for col in self.columns for elem in row[col]]
        self.input_size = len(input)
        self.output_size = 1


    def __len__(self) -> int:
        """Returns the length of the Dataset in a lazy fashion.

        Returns:
            - int : Length of the Dataset.
        """
        return self.dataframe.select(pl.count()).collect().item()


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
        input = [elem for col in self.columns for elem in row[col]]

        # Get the target
        target = row[self.target]

        return torch.tensor(input, dtype=torch.float), torch.tensor([target], dtype=torch.float)



# =====================================================
#
#                DATAMODULE DEFINITION
#
# =====================================================
    
class DataModule(LightningDataModule):
    """A custom Lightning Data Module to handle a Pytorch Dataset.

    Args:
        - path (str): The str of the path towards the data. It will be passed as argument to the CustomDataset.
        - columns (list[str]): The list of columns to use as input. It will be passed as argument to the CustomDataset.
        - target (str): The name of the column to use as target. It will be passed as argument to the CustomDataset.
        - batch_size (int): The size of a batch. Default 128.
        - num_workers (int): The number of workers. Setting it to 0 means that the data will be
                            loaded in the main process. Default 0.
        - train_size (int | float): The size of the train data. Default 0.7.
        - test_size (int | float): The size of the test data. Default 0.15.
        - val_size (int | float): The size of the val data. Default 0.15.

    Attributes:
        The self.<same_name_of_args> version of the arguments documented above.
        The only difference is that self.path is stored as Path and not as str.
    """
    def __init__(self,
                 path : str,
                 columns: list[str],
                 target: str,
                 batch_size: int = 128,
                 num_workers: int = 0,
                 train_size: int | float = 0.7,
                 test_size: int | float = 0.15,
                 val_size: int | float = 0.15) -> None:
        super().__init__()

        self.path = Path(path)
        self.columns = columns
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size


    def prepare_data(self) -> None:
        """This function will extracts the examples parquet files from the example_data.zip file.

        Returns:
            - None
        """
        from zipfile import ZipFile

        current = Path('.')
        zip_path = current / 'data/example_data.zip'

        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(zip_path.parent)

        return None
    

    def setup(self,
              stage: str = None) -> None:
        """This function setups a CustomDataset for our data.

        Returns:
            - None.
        """
        data = CustomDataset(self.path, self.columns, self.target)
        self.input_size = data.input_size
        self.output_size = data.output_size
        self.train_data, self.test_data, self.val_data = random_split(data, [self.train_size, self.test_size, self.val_size])
        return None


    def train_dataloader(self) -> DataLoader:
        """The function returns the train DataLoader.

        Returns:
            -  DataLoader : The train DataLoader.
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def test_dataloader(self) -> DataLoader:
        """The function returns the test DataLoader.

        Returns:
            -  DataLoader : The test DataLoader.
        """
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def val_dataloader(self) -> DataLoader:
        """The function returns the validation DataLoader.

        Returns:
            -  DataLoader : The validation DataLoader.
        """
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



def main() -> None:
    """The main script loop in which we perform some sanity tests.
    """
    print("Start performing sanity tests...")
    print()

    print("Running first test...", end='\t')
    # Setting inputs
    path = "./data/example_n1000_dim10_tanh_anchors10.parquet"
    columns = ['x_i', 'a_j']
    target = 'r_ij'

    data = DataModule(path=path, columns=columns, target=target)

    data.prepare_data()
    data.setup()
    next(iter(data.train_dataloader()))
    next(iter(data.test_dataloader()))
    next(iter(data.val_dataloader()))

    print('[Passed]')

    return None


if __name__ == "__main__":
    main()
