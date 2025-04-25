"""
This python module computes a simulation of a transmitter communicating with a receiver via a MIMO channel.
"""

# Add root to the path
import sys
import typing
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

if typing.TYPE_CHECKING:
    import torch

import wandb
import hydra
import polars as pl
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from torch.utils.data import TensorDataset, DataLoader

from src.linear_models import Baseline
from src.neural_models import Classifier
from src.utils import complex_gaussian_matrix
from src.download_utils import download_zip_from_gdrive
from src.datamodules import DataModule


def setup(
    models_path: Path,
) -> None:
    """Setup the repository:
    - downloading classifiers models.

    Args:
        models_path : Path
            The path to the models
    """
    print()
    print('Start setup procedure...')

    print()
    print('Check for the classifiers model availability...')
    # Download the classifiers if needed
    # Get from the .env file the zip file Google Drive ID
    id = dotenv_values()['CLASSIFIER_ID']
    download_zip_from_gdrive(id=id, name='classifiers', path=str(models_path))

    print()
    print('All done.')
    print()
    return None


# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================


@hydra.main(
    config_path='../.conf/hydra/baseline',
    config_name='train_baseline',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """The main loop."""

    # Only square channel if cfg.communication.square is set to true
    if cfg.communication.square and (
        cfg.communication.antennas_receiver
        != cfg.communication.antennas_transmitter
    ):
        return None

    # Define some usefull paths
    CURRENT: Path = Path('.')
    MODEL_PATH: Path = CURRENT / 'models'
    RESULTS_PATH: Path = CURRENT / 'results/baseline'

    # Create results directory
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    # Define some variables
    trainer: Trainer = Trainer(
        inference_mode=True,
        enable_progress_bar=False,
        logger=False,
        accelerator=cfg.device,
    )

    # Setup procedure
    setup(models_path=MODEL_PATH)

    # Convert DictConfig to a standard dictionary before passing to wandb
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    # Initialize W&B and log config
    wandb.init(
        project=cfg.wandb.project,
        name=f'{cfg.seed}_{cfg.communication.awareness}_{cfg.communication.antennas_receiver}_{cfg.communication.antennas_transmitter}_{cfg.communication.snr}_{cfg.simulation}_{cfg.datamodule.dataset}_{cfg.datamodule.train_label_size}_{cfg.strategy}',
        id=f'{cfg.seed}_{cfg.communication.awareness}_{cfg.communication.antennas_receiver}_{cfg.communication.antennas_transmitter}_{cfg.communication.snr}_{cfg.simulation}_{cfg.datamodule.dataset}_{cfg.datamodule.train_label_size}_{cfg.strategy}',
        config=wandb_config,
    )

    # Setting the seed
    seed_everything(cfg.seed, workers=True)

    # Channel Initialization
    channel_matrix: torch.Tensor = complex_gaussian_matrix(
        0,
        1,
        (
            cfg.communication.antennas_receiver,
            cfg.communication.antennas_transmitter,
        ),
    )

    # =============================================================
    #                 Datamodule Initialization
    # =============================================================
    datamodule: DataModule = DataModule(
        dataset=cfg.datamodule.dataset,
        tx_enc=cfg.transmitter.model,
        rx_enc=cfg.receiver.model,
        train_label_size=cfg.datamodule.train_label_size,
        method=cfg.datamodule.method,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.workers,
    )

    # Prepare and setup the data
    datamodule.prepare_data()
    datamodule.setup()

    # =============================================================
    #                 Classifier Initialization
    # =============================================================
    # Define the path towards the classifier
    clf_path: Path = (
        MODEL_PATH
        / f'classifiers/{cfg.datamodule.dataset}/{cfg.receiver.model}/seed_{cfg.seed}.ckpt'
    )

    # Load the classifier model
    clf = Classifier.load_from_checkpoint(clf_path)
    clf.eval()

    # =============================================================
    #                 Define the Linear Optimizer
    # =============================================================
    opt: Baseline = Baseline(
        input_dim=datamodule.input_size,
        output_dim=datamodule.output_size,
        channel_matrix=channel_matrix,
        snr=cfg.communication.snr,
        channel_usage=cfg.communication.channel_usage,
        typology=cfg.typology,
    )

    # Fit the linear optimizer
    opt.fit(
        input=datamodule.train_data.z_tx,
        output=datamodule.train_data.z_rx,
    )

    # =============================================================
    #                 Evaluate over the test set
    # =============================================================
    # Get the z_psi_hat
    z_psi_hat = opt.transform(datamodule.test_data.z_tx)

    # Get the predictions using as input the z_psi_hat
    dataloader = DataLoader(
        TensorDataset(z_psi_hat, datamodule.test_data.labels),
        batch_size=cfg.datamodule.batch_size,
    )
    clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]

    # =============================================================
    #                     Save the results
    # =============================================================
    pl.DataFrame(
        {
            'Dataset': cfg.datamodule.dataset,
            'Training Label Size': cfg.datamodule.train_label_size,
            'Seed': cfg.seed,
            'Antennas Transmitter': cfg.communication.antennas_transmitter,
            'Antennas Receiver': cfg.communication.antennas_receiver,
            'Sparsity': None,
            'Lambda': None,
            'SNR': cfg.communication.snr,
            'Awareness': cfg.communication.awareness,
            'Cost': cfg.px_cost,
            'FLOPs': None,
            'Accuracy': clf_metrics['test/acc_epoch'],
            'Alignment Loss': opt.eval(
                datamodule.test_data.z_tx, datamodule.test_data.z_rx
            ),
            'Classifier Loss': clf_metrics['test/loss_epoch'],
            'Receiver Model': cfg.receiver.model,
            'Transmitter Model': cfg.transmitter.model,
            'Case': 'Baseline ' + cfg.strategy,
            'Latent Real Dim': 2 * datamodule.input_size
            if cfg.strategy == 'Top-K'
            else datamodule.input_size,
            'Latent Complex Dim': 2 * ((datamodule.input_size + 1) // 2)
            if cfg.strategy == 'Top-K'
            else (datamodule.input_size + 1) // 2,
            'Simulation': cfg.simulation,
        }
    ).write_parquet(
        RESULTS_PATH
        / f'{cfg.seed}_{cfg.communication.antennas_transmitter}_{cfg.communication.antennas_receiver}_{cfg.communication.snr}_{cfg.datamodule.dataset}_{cfg.datamodule.train_label_size}_{cfg.communication.awareness}_{cfg.strategy}_{cfg.simulation}.parquet'
    )

    # Closing W&B
    wandb.finish()
    return None


if __name__ == '__main__':
    main()
