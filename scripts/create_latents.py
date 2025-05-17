# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "hydra-core",
#     "omegaconf",
#     "pytorch-lightning",
#     "timm",
#     "torch",
#     "tqdm",
# ]
# ///
"""A script to collect the needed latents."""

import hydra
import random
from tqdm import tqdm
from pathlib import Path
from functools import partial
from omegaconf import DictConfig

from datasets import load_dataset, Dataset

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# =============================================================
#
#                 DEFINE SOME SCRIPT METHODS
#
# =============================================================


def get_dataset(
    dataset: str,
    split: str,
    perc: float = 1.0,
    val_perc: float = 0.15,
    seed: int = 42,
) -> Dataset | tuple[Dataset, Dataset]:
    """A function to retrieve a dataset from the Datasets library.

    Args:
        dataset : str
            The name of the dataset.
        split : str
            Which set we want to retrieve, possible values in ['train', 'test'].
        perc : float
            The percentage of the original dataset to retrieve. Default 1.0.
        val_perc : float
            The percentage of the train set to use as validation, used if split='train'. Default 0.15.
        seed : int
            The seed for the random shuffle. Default 42.

    Returns:
        dataset : Dataset | tuple[Dataset, Dataset]
            The wanted Dataset.
    """
    seed_everything(seed)
    assert 0 < perc <= 1
    assert 0 < val_perc < 1
    dataset = load_dataset(dataset)[split]

    # If the split is 'train', further split into train and validation sets
    match split:
        case 'train':
            # Shuffle and select a random subset for the entire training set
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[: int(len(indices) * perc)]

            # Calculate lengths and indices for train and validation subsets
            val_len = int(len(indices) * val_perc)
            train_len = len(indices) - val_len

            train_indices = indices[:train_len]
            val_indices = indices[train_len:]

            # Create train and validation subsets
            train_dataset = dataset.select(train_indices)
            val_dataset = dataset.select(val_indices)

            dataset = (train_dataset, val_dataset)
        case 'test':
            # For other splits, you can apply the original random subset logic
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[: int(len(indices) * perc)]
            dataset = dataset.select(indices)
        case _:
            raise Exception('The passed "split" parameter is not in allowed.')

    return dataset


def load_model(model_name: str) -> Module:
    """A function to load a timm model.

    Args:
        model_name : str
            The name of the model.

    Returns:
        model : Module
            The specific model.
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    return model.requires_grad_(False).eval()


@torch.no_grad()
def call_model(
    batch: dict[str, torch.Tensor],
    model: Module,
    device: str,
) -> dict[str, torch.Tensor]:
    """Get the encodings.

    Args:
        batch : dict[str, torch.Tensor]
        model : Module
            The model.
        device : str.
            The device on for pytorch.

    Returns:
        dict[str, torch.Tensor]
            The encodings.
    """
    sample_encodings = model(batch['encoding'].to(device))
    return {'hidden': sample_encodings}


def get_latents(
    dataloader: DataLoader,
    model: Module,
    split: str,
    device: str,
) -> dict[str, torch.Tensor]:
    """A function to get the latents of a model.

    Args:
        dataloader : Dataloader
            The dataloader which contains the dataset examples.
        model : Module
            The model.
        split : str
            The split to consider. Available 'train', 'val' and 'test'.
        device : str
            The device for pytorch.

    Returns:
        dict[str, torch.Tensor]
            A dictionary containing the absolute version of the latents and the respective labels.
    """
    absolute_latents: list = []
    labels: list = []

    model = model.to(device)
    for batch in tqdm(dataloader, desc=f'[{split}] Computing latents'):
        with torch.no_grad():
            model_out = call_model(
                batch=batch,
                model=model,
                device=device,
            )
            absolute_latents.append(model_out['hidden'].cpu())
            labels.append(batch['label'].cpu())

    absolute_latents: torch.Tensor = torch.cat(absolute_latents, dim=0).cpu()
    labels: torch.Tensor = torch.cat(labels, dim=0).cpu()

    model = model.cpu()
    return {
        'absolute': absolute_latents,
        'labels': labels,
    }


def collate_fn(batch, feature_extractor, transform):
    return {
        'encoding': torch.stack(
            [transform(sample['img'].convert('RGB')) for sample in batch],
            dim=0,
        ),
        'label': torch.tensor([sample['label'] for sample in batch]),
    }


def encode_latents(
    models: list[str],
    dataset: Dataset,
    split: str,
    device: str,
    path: Path = None,
) -> None:
    """A function to handle the latents encoding.

    Args:
        models : list[str]
            The list of model names.
        dataset : Dataset
            The dataset.
        split : str
            The split to consider. Available 'train', 'val' and 'test'.
        device : str
            The device to use for torch.
        path : Path
            The path where to sace the encodings, if set to None the encodings are note saved. Default None.

    Returns:
        None
    """
    assert split in {'train', 'val', 'test'}, (
        'The passed split is not in the available ones.'
    )

    model2latents = {}

    for model_name in models:
        # Load the transformer model
        model = load_model(model_name=model_name)

        # Create a transform for the data based on the model's requirements
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

        # Process the main dataset
        dataset_latents_output = get_latents(
            dataloader=DataLoader(
                dataset,
                num_workers=0,
                pin_memory=True,
                collate_fn=partial(
                    collate_fn, feature_extractor=None, transform=transform
                ),
                batch_size=32,
            ),
            split=f'{split}/{model_name}',
            model=model,
            device=device,
        )

        # Store the latents and labels
        model2latents[model_name] = {
            **dataset_latents_output,
        }

        # Save latents and labels if caching is enabled
        if path is not None:
            print(f'Saving latents and labels for {model_name}...')
            model_path: Path = (
                path / split / f'{model_name.replace("/", "-")}.pt'
            )
            model_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model2latents[model_name], model_path)
    return None


# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================


@hydra.main(
    config_path='../.conf/hydra/dataset',
    config_name='create_latents',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """The main loop."""

    # Define some paths
    CURRENT: Path = Path('.')
    DATA_PATH: Path = CURRENT / 'data'
    LATENTS_DIR = DATA_PATH / f'latents/{cfg.dataset}'
    LATENTS_DIR.mkdir(exist_ok=True, parents=True)

    # Get the train and validation set
    train_dataset, val_dataset = get_dataset(
        dataset=cfg.dataset,
        split='train',
        perc=cfg.perc,
        val_perc=cfg.val_perc,
        seed=cfg.seed,
    )
    print(f'train: {len(train_dataset)}', f'validation: {len(val_dataset)}')

    # Get the test set
    test_dataset = get_dataset(
        dataset=cfg.dataset,
        split='test',
        perc=cfg.perc,
        seed=cfg.seed,
    )
    print(f'test: {len(test_dataset)}')

    # Seed everything
    seed_everything(cfg.seed)

    # =============================================================
    #                      TRAIN LATENTS
    # =============================================================
    encode_latents(
        models=cfg.models,
        dataset=train_dataset,
        split='train',
        device=cfg.device,
        path=LATENTS_DIR,
    )

    # =============================================================
    #                      VAL LATENTS
    # =============================================================
    encode_latents(
        models=cfg.models,
        dataset=val_dataset,
        split='val',
        device=cfg.device,
        path=LATENTS_DIR,
    )

    # =============================================================
    #                      TEST LATENTS
    # =============================================================
    encode_latents(
        models=cfg.models,
        dataset=test_dataset,
        split='test',
        device=cfg.device,
        path=LATENTS_DIR,
    )

    return None


if __name__ == '__main__':
    main()
