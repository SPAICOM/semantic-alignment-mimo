"""In this python module we define class that handles the dataset:
    - CustomDataset: a custom Pytorch Dataset.
    - CustomDatasetRelativeDecoder: a custom Pytorch Dataset for decoding from a relative representation to an absolute one.
    - DataModule: a Pytorch Lightning Data Module.
"""

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


# =====================================================
#
#                 DATASETS DEFINITION
#
# =====================================================

class CustomDataset(Dataset):
    """A custom implementation of a Pytorch Dataset.

    Args:
        encoder_path (Path): The path to the encoder.
        decider_path (Path): The path to the decoder.
        num_anchors (int): The number of anchors to use.
        case (str): The input case. Choose between 'rel', 'abs' or 'abs_anch'.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
        self.z (torch.Tensor): The absolute representation of the Dataset encoder side.
        self.anchors (torch.Tensor): The absolute representation of the anchors encoder side.
        self.r_encoder (torch.Tensor): The relative representation of the Dataset encoder side.
        self.r_decoder (torch.Tensor): The relative representation of the Dataset decoder side.
        self.input_size (int): The size of the input of the network.
        self.output_size (int): The size of the output of the network.
    """
    def __init__(self,
                 encoder_path: Path,
                 decoder_path: Path,
                 num_anchors: int,
                 case: str):
        self.encoder_path: Path = encoder_path
        self.decoder_path: Path = decoder_path
        self.num_anchors: int = num_anchors
        self.case: str = case

        assert self.case in ['rel', 'abs', 'abs_anch'], "Wrong case passed, choose between 'rel', 'abs' or 'abs_anch'."
        
        # =================================================
        #                 Encoder Stuff
        # =================================================
        encoder_blob = torch.load(self.encoder_path)

        # Retrieve the absolute representation from the encoder
        self.z = encoder_blob['absolute']

        # Retrieve the anchors from the encoder
        self.anchors = encoder_blob['anchors_latents']

        assert self.z.shape[-1] == self.anchors.shape[-1], "The dimension of the anchors and of the absolute representation must be equal."
        assert num_anchors <= len(self.anchors), "The passed number of anchors exceed the total number of available anchors."
        
        # Select the wanted anchors
        self.anchors = self.anchors[:num_anchors]

        # Retriece the relative representation from the encoder
        self.r_econder = encoder_blob['relative'][:, :self.num_anchors]

        del encoder_blob

        # =================================================
        #                 Decoder Stuff
        # =================================================
        decoder_blob = torch.load(self.decoder_path)

        # Retrieve the relative representation from the decoder
        self.r_decoder = decoder_blob['relative'][:, :self.num_anchors]

        del decoder_blob
        
        # =================================================
        #         Get the input and the output size
        # =================================================
        # When the input is only the absolute representation
        if self.case == 'abs':
            self.input_size = self.z.shape[-1]

        # When the input is both the absolute representation and the anchors
        elif self.case == 'abs_anch':
            self.input_size = self.z.shape[-1] + self.anchors.shape[-1]*self.num_anchors

        # When the input is only the relative representation
        elif self.case == 'rel':
            self.input_size = self.r_econder.shape[-1]
            
        self.output_size = self.num_anchors


    def __len__(self) -> int:
        """Returns the length of the Dataset.

        Returns:
            int : Length of the Dataset.
        """
        return len(self.z)


    def __getitem__(self,
                    idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns in a torch.Tensor format the input and the target.

        Args:
            idx (int): The index of the wanted row.

        Returns:
            tuple[torch.Tensor, torch.Tensor] : The inputs and target as a tuple of tensors.
        """
        # If input is only the absolute representation
        if self.case == 'abs':
            # Get the absolute representation of element idx
            input = self.z[idx]

        # If the input is both the absolute representation and the anchors
        elif self.case == 'abs_anch':
            # Get the absolute representation of element idx
            z_i = self.z[idx]

            # Define an input of the shape [z_i, a_1, ..., a_n]
            input = torch.cat((z_i.unsqueeze(0), self.anchors), 0)
            input = torch.flatten(input)

        # If the input is only the relative representation
        elif self.case == 'rel':
            input = self.r_econder[idx]
        
        # Get the relative representation of element idx
        r_i = self.r_decoder[idx]

        return input, r_i


class CustomDatasetRelativeDecoder(Dataset):
    """A custom implementation of a Pytorch Dataset.

    Args:
        path (Path): The path to the data.
        num_anchors (int): The number of anchors to use.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
        self.z (torch.Tensor): The absolute representation of the Dataset.
        self.anchors (torch.Tensor): The absolute representation of the anchors.
        self.r (torch.Tensor): The relative representation of the Dataset.
        self.input_size (int): The size of the input of the network.
        self.output_size (int): The size of the output of the network.
    """
    def __init__(self,
                 path: Path,
                 num_anchors: int):
        self.path: Path = path
        self.num_anchors: int = num_anchors
        
        # =================================================
        #                 Get the Data
        # =================================================
        decoder_blob = torch.load(self.path)

        # Retrieve the absolute representation from the decoder
        self.z = decoder_blob['absolute']

        # Retrieve the anchors from the decoder
        self.anchors = decoder_blob['anchors_latents']

        assert self.z.shape[-1] == self.anchors.shape[-1], "The dimension of the anchors and of the absolute representation must be equal."
        assert num_anchors <= len(self.anchors), "The passed number of anchors exceed the total number of available anchors."
        
        # Select the wanted anchors
        self.anchors = self.anchors[:num_anchors]

        # Retrieve the relative representation from the decoder
        self.r = decoder_blob['relative'][:, :self.num_anchors]

        del decoder_blob

        # =================================================
        #         Get the input and the output size
        # =================================================
        self.input_size = self.r.shape[-1]
        self.output_size = self.z.shape[-1]


    def __len__(self) -> int:
        """Returns the length of the Dataset.

        Returns:
            int : Length of the Dataset.
        """
        return len(self.r)


    def __getitem__(self,
                    idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns in a torch.Tensor format the input and the target.

        Args:
            idx (int): The index of the wanted row.

        Returns:
            tuple[torch.Tensor, torch.Tensor] : The inputs and target as a tuple of tensors.
        """
        # Get the relative representation of the element at idx
        r_i = self.r[idx]

        # Get the absolute representation of the element at idx
        z_i = self.z[idx]
        
        return r_i, z_i


# =====================================================
#
#                DATAMODULES DEFINITION
#
# =====================================================
    
class DataModule(LightningDataModule):
    """A custom Lightning Data Module to handle a Pytorch Dataset.

    Args:
        dataset (str): The name of the dataset.
        encoder (str): The name of the encoder.
        decoder (str): The name of the decoder.
        num_anchors (int): The number of anchors to use.
        case (str): The case argument of the CustomDataset.
        batch_size (int): The size of a batch. Default 128.
        num_workers (int): The number of workers. Setting it to 0 means that the data will be
                            loaded in the main process. Default 0.
        train_size (int | float): The size of the train data. Default 0.7.
        test_size (int | float): The size of the test data. Default 0.15.
        val_size (int | float): The size of the val data. Default 0.15.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
    """
    def __init__(self,
                 dataset: str,
                 encoder: str,
                 decoder: str,
                 num_anchors: int,
                 case: str,
                 batch_size: int = 128,
                 num_workers: int = 0,
                 train_size: int | float = 0.7,
                 test_size: int | float = 0.15,
                 val_size: int | float = 0.15) -> None:
        super().__init__()

        self.dataset: str = dataset
        self.encoder: str = encoder
        self.decoder: str = decoder
        self.case: str = case
        self.num_anchors: int = num_anchors
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.train_size: int | float = train_size
        self.test_size: int | float = test_size
        self.val_size: int | float = val_size


    def prepare_data(self) -> None:
        """This function prepares the dataset (Download and Unzip).

        Returns:
            None
        """
        from gdown import download
        from zipfile import ZipFile
        from dotenv import dotenv_values

        CURRENT = Path('.')
        DATA_DIR = CURRENT / 'data'
        ZIP_PATH = DATA_DIR / 'latents.zip'
        DIR_PATH = DATA_DIR / 'latents/'

        # Make sure that DATA_DIR exists
        DATA_DIR.mkdir(exist_ok=True)

        # Check if the zip file is already in the path
        if not ZIP_PATH.exists():
            # Get from the .env file the zip file Google Drive ID
            ID = dotenv_values()['ID']

            # Download the zip file
            download(id=ID, output=str(ZIP_PATH))
            
        # Check if the directory exists
        if not DIR_PATH.is_dir():
            # Unzip the zip file
            with ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(ZIP_PATH.parent)

        return None
    

    def setup(self,
              stage: str = None) -> None:
        """This function setups a CustomDataset for our data.

        Args:
            - stage (str): The stage of the setup. Default None.

        Returns:
            - None.
        """
        CURRENT = Path('.')
        GENERAL_PATH: Path = CURRENT / 'data/latents' / self.dataset

        self.train_data = CustomDataset(encoder_path=GENERAL_PATH / 'train' / f'{self.encoder}.pt',
                                        decoder_path=GENERAL_PATH / 'train' / f'{self.decoder}.pt',
                                        num_anchors=self.num_anchors,
                                        case=self.case)
        self.test_data = CustomDataset(encoder_path=GENERAL_PATH / 'test' / f'{self.encoder}.pt',
                                       decoder_path=GENERAL_PATH / 'test' / f'{self.decoder}.pt',
                                       num_anchors=self.num_anchors,
                                       case=self.case)
        self.val_data = CustomDataset(encoder_path=GENERAL_PATH / 'val' / f'{self.encoder}.pt',
                                      decoder_path=GENERAL_PATH / 'val' / f'{self.decoder}.pt',
                                      num_anchors=self.num_anchors,
                                      case=self.case)

        assert self.train_data.input_size == self.test_data.input_size and self.train_data.input_size == self.val_data.input_size, "Input size must match between train, test and val data."
        assert self.train_data.output_size == self.test_data.output_size and self.train_data.output_size == self.val_data.output_size, "Output size must match between train, test and val data."

        self.input_size = self.train_data.input_size
        self.output_size = self.train_data.output_size
        return None


    def train_dataloader(self) -> DataLoader:
        """The function returns the train DataLoader.

        Returns:
            DataLoader : The train DataLoader.
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def test_dataloader(self) -> DataLoader:
        """The function returns the test DataLoader.

        Returns:
            DataLoader : The test DataLoader.
        """
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def val_dataloader(self) -> DataLoader:
        """The function returns the validation DataLoader.

        Returns:
            DataLoader : The validation DataLoader.
        """
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



class DataModuleRelativeDecoder(LightningDataModule):
    """A custom Lightning Data Module to handle a Pytorch Dataset.

    Args:
        dataset (str): The name of the dataset.
        decoder (str): The name of the decoder.
        num_anchors (int): The number of anchors to use.
        batch_size (int): The size of a batch. Default 128.
        num_workers (int): The number of workers. Setting it to 0 means that the data will be
                            loaded in the main process. Default 0.
        train_size (int | float): The size of the train data. Default 0.7.
        test_size (int | float): The size of the test data. Default 0.15.
        val_size (int | float): The size of the val data. Default 0.15.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
    """
    def __init__(self,
                 dataset: str,
                 decoder: str,
                 num_anchors: int,
                 batch_size: int = 128,
                 num_workers: int = 0,
                 train_size: int | float = 0.7,
                 test_size: int | float = 0.15,
                 val_size: int | float = 0.15) -> None:
        super().__init__()

        self.dataset: str = dataset
        self.decoder: str = decoder
        self.num_anchors: int = num_anchors
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.train_size: int | float = train_size
        self.test_size: int | float = test_size
        self.val_size: int | float = val_size


    def prepare_data(self) -> None:
        """This function prepare the dataset (Download and Unzip).

        Returns:
            None
        """
        from gdown import download
        from zipfile import ZipFile
        from dotenv import dotenv_values

        CURRENT = Path('.')
        DATA_DIR = CURRENT / 'data'
        ZIP_PATH = DATA_DIR / 'latents.zip'
        DIR_PATH = DATA_DIR / 'latents/'

        # Make sure that DATA_DIR exists
        DATA_DIR.mkdir(exist_ok=True)

        # Check if the zip file is already in the path
        if not ZIP_PATH.exists():
            # Get from the .env file the zip file Google Drive ID
            ID = dotenv_values()['ID']

            # Download the zip file
            download(id=ID, output=str(ZIP_PATH))
            
        # Check if the directory exists
        if not DIR_PATH.is_dir():
            # Unzip the zip file
            with ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(ZIP_PATH.parent)

        return None
    

    def setup(self,
              stage: str = None) -> None:
        """This function setups a CustomDatasetRelativeDecoder for our data.

        Args:
            stage (str): The stage of the setup. Default None.

        Returns:
            None.
        """
        CURRENT = Path('.')
        GENERAL_PATH: Path = CURRENT / 'data/latents' / self.dataset

        self.train_data = CustomDatasetRelativeDecoder(path=GENERAL_PATH / 'train' / f'{self.decoder}.pt',
                                                       num_anchors=self.num_anchors)
        self.test_data = CustomDatasetRelativeDecoder(path=GENERAL_PATH / 'test' / f'{self.decoder}.pt',
                                                      num_anchors=self.num_anchors)
        self.val_data = CustomDatasetRelativeDecoder(path=GENERAL_PATH / 'val' / f'{self.decoder}.pt',
                                                     num_anchors=self.num_anchors)

        assert self.train_data.input_size == self.test_data.input_size and self.train_data.input_size == self.val_data.input_size, "Input size must match between train, test and val data."
        assert self.train_data.output_size == self.test_data.output_size and self.train_data.output_size == self.val_data.output_size, "Output size must match between train, test and val data."

        self.input_size = self.train_data.input_size
        self.output_size = self.train_data.output_size
        return None


    def train_dataloader(self) -> DataLoader:
        """The function returns the train DataLoader.

        Returns:
            DataLoader : The train DataLoader.
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def test_dataloader(self) -> DataLoader:
        """The function returns the test DataLoader.

        Returns:
            DataLoader : The test DataLoader.
        """
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def val_dataloader(self) -> DataLoader:
        """The function returns the validation DataLoader.

        Returns:
            DataLoader : The validation DataLoader.
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
    num_anchors = 1024
    case = 'rel'
    
    data = DataModule(dataset=dataset,
                      encoder=encoder,
                      decoder=decoder,
                      num_anchors=num_anchors,
                      case=case)

    data.prepare_data()
    data.setup()
    next(iter(data.train_dataloader()))
    next(iter(data.test_dataloader()))
    next(iter(data.val_dataloader()))

    print('[Passed]')

    print("Running second test...", end='\t')
    
    # Setting inputs
    dataset = 'cifar100'
    encoder = 'mobilenetv3_small_100'
    decoder = 'rexnet_100'
    num_anchors = 1024
    
    data = DataModuleRelativeDecoder(dataset=dataset,
                                     decoder=decoder,
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
