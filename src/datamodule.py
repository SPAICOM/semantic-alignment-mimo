"""In this python module we define class that handles the dataset:
    - CustomDataset: a custom Pytorch Dataset.
    - DataModule: a Pytorch Lightning Data Module.
"""

import torch
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
        - encoder_path (Path): The path to the encoder.
        - decider_path (Path): The path to the decoder.
        - num_anchors (int): The number of anchors to use.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
    """
    def __init__(self,
                 encoder_path: Path,
                 decoder_path: Path,
                 num_anchors: int):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.num_anchors = num_anchors

        # =================================================
        #                 Encoder Stuff
        # =================================================
        encoder_blob = torch.load(self.encoder_path)

        # Retrieve the anchors
        self.anchors = encoder_blob['anchors_latents']

        assert num_anchors <= len(self.anchors), "The passed number of anchors exceed the total number of available anchors."
        
        # Select the wanted anchors
        self.anchors = self.anchors[:num_anchors]

        # Retrieve the absolute representation from the encoder
        self.z = encoder_blob['absolute']

        del encoder_blob

        # =================================================
        #                 Decoder Stuff
        # =================================================
        decoder_blob = torch.load(self.decoder_path)

        # Retrieve the relative representation from the decoder
        self.r = decoder_blob['relative'][:, :self.num_anchors]

        del decoder_blob
        
        # =================================================
        #         Get the input and the output size
        # =================================================
        assert self.z.shape[-1] == self.anchors.shape[-1], "The dimension of the anchors and of the absolute representation must be equal."
        self.input_size = self.z.shape[-1] + self.anchors.shape[-1]*self.num_anchors
        self.output_size = self.num_anchors


    def __len__(self) -> int:
        """Returns the length of the Dataset.

        Returns:
            - int : Length of the Dataset.
        """
        return len(self.z)


    def __getitem__(self,
                    idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns in a torch.Tensor format the input and the target.

        Args:
            - idx (int): The index of the wanted row.

        Returns:
            - tuple[torch.Tensor, torch.Tensor] : The inputs and target as a tuple of tensors.
        """
        # Get the absolute representation of element idx
        z_i = self.z[idx]

        # Define an input of the shape [z_i, a_1, ..., a_n]
        input = torch.cat((z_i.unsqueeze(0), self.anchors), 0)
        
        # Get the relative representation of element idx
        r_i = self.r[idx]

        return torch.flatten(input), r_i



# =====================================================
#
#                DATAMODULE DEFINITION
#
# =====================================================
    
class DataModule(LightningDataModule):
    """A custom Lightning Data Module to handle a Pytorch Dataset.

    Args:
        - dataset (str): The name of the dataset.
        - encoder (str): The name of the encoder.
        - decoder (str): The name of the decoder.
        - split (str): The name of the split. Choose between ['train', 'test', 'val'].
        - num_anchors (int): The number of anchors to use.
        - batch_size (int): The size of a batch. Default 128.
        - num_workers (int): The number of workers. Setting it to 0 means that the data will be
                            loaded in the main process. Default 0.
        - train_size (int | float): The size of the train data. Default 0.7.
        - test_size (int | float): The size of the test data. Default 0.15.
        - val_size (int | float): The size of the val data. Default 0.15.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
        - self.path_encoder (Path): The path to the encoder.
        - self.path_decoder (Path): The path to the decoder.
    """
    def __init__(self,
                 dataset: str,
                 encoder: str,
                 decoder: str,
                 split: str,
                 num_anchors: int,
                 batch_size: int = 128,
                 num_workers: int = 0,
                 train_size: int | float = 0.7,
                 test_size: int | float = 0.15,
                 val_size: int | float = 0.15) -> None:
        super().__init__()

        CURRENT = Path('.')
        GENERAL_PATH = CURRENT / 'data/latents' / dataset / split 

        self.path_encoder = GENERAL_PATH / f'{encoder}.pt'
        self.path_decoder = GENERAL_PATH / f'{decoder}.pt'
        self.dataset = dataset
        self.encoder = encoder
        self.decoder = decoder
        self.split = split
        self.num_anchors = num_anchors
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size


    def prepare_data(self) -> None:
        """This function prepare the dataset (Download and Unzip).

        Returns:
            - None
        """
        from gdown import download
        from zipfile import ZipFile
        from dotenv import dotenv_values

        CURRENT = Path('.')
        ZIP_PATH = CURRENT / 'data/latents.zip'
        DIR_PATH = CURRENT / 'data/latents/'

        # Check if the zip file is already in the path
        if not ZIP_PATH.exists():
            # Get from the .env file the zip file Google Drive ID
            ID = dotenv_values()['ID']

            # Download the zip file
            download(id=ID, output=str(ZIP_PATH))
            
        if not DIR_PATH.is_dir():
            # Unzip the zip file
            with ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(ZIP_PATH.parent)

        return None
    

    def setup(self,
              stage: str = None) -> None:
        """This function setups a CustomDataset for our data.

        Returns:
            - None.
        """
        data = CustomDataset(self.path_encoder, self.path_decoder, self.num_anchors)
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
    dataset = 'cifar100'
    encoder = 'mobilenetv3_small_100'
    decoder = 'rexnet_100'
    split = 'test'
    num_anchors = 1024
    
    data = DataModule(dataset=dataset,
                      encoder=encoder,
                      decoder=decoder,
                      split=split,
                      num_anchors=num_anchors)

    data.prepare_data()
    data.setup()
    next(iter(data.train_dataloader()))
    next(iter(data.test_dataloader()))
    next(iter(data.val_dataloader()))

    print('[Passed]')

    return None


if __name__ == "__main__":
    main()
