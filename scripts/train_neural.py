"""
This python module handles the training of the neural model.
"""

# Add root to the path
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import torch
import wandb
import hydra
import polars as pl
from dotenv import dotenv_values
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    BatchSizeFinder,
)

from src.datamodules import DataModule
from src.utils import complex_gaussian_matrix, remove_non_empty_dir
from src.resource_profiler import count_nonzero_weights, neural_sparse_flops
from src.neural_models import NeuralModel, Classifier
from src.download_utils import download_zip_from_gdrive


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
    config_path='../.conf/hydra/neural',
    config_name='train_neural',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """The main script loop."""
    # Only square channel if cfg.communication.square is set to true
    if cfg.communication.square and (
        cfg.communication.antennas_receiver
        != cfg.communication.antennas_transmitter
    ):
        return None

    # Define some usefull paths
    CURRENT: Path = Path('.')
    MODEL_PATH: Path = CURRENT / 'models'
    RESULTS_PATH: Path = CURRENT / 'results/neural'

    # Create results directory
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    # Setup procedure
    setup(models_path=MODEL_PATH)

    # Callbacks definition
    callbacks = [
        LearningRateMonitor(logging_interval='step', log_momentum=True),
        ModelCheckpoint(monitor='valid/loss_epoch', save_top_k=1, mode='min'),
        BatchSizeFinder(mode='binsearch', max_trials=8),
        EarlyStopping(monitor='valid/loss_epoch', patience=10),
    ]

    # Convert DictConfig to a standard dictionary before passing to wandb
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    # W&B login and Logger intialization
    wandb.login()
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=f'{cfg.seed}_{cfg.communication.awareness}_{cfg.communication.antennas_receiver}_{cfg.communication.antennas_transmitter}_{cfg.communication.snr}_{cfg.simulation}_{cfg.datamodule.dataset}_{cfg.datamodule.grouping}_{cfg.datamodule.method}_{cfg.datamodule.train_label_size}_{cfg.model.lmb}',
        id=f'{cfg.seed}_{cfg.communication.awareness}_{cfg.communication.antennas_receiver}_{cfg.communication.antennas_transmitter}_{cfg.communication.snr}_{cfg.simulation}_{cfg.datamodule.dataset}_{cfg.datamodule.grouping}_{cfg.datamodule.method}_{cfg.datamodule.train_label_size}_{cfg.model.lmb}',
        config=wandb_config,
        log_model=cfg.wandb.log_model,
    )

    # Trainer Definition
    trainer = Trainer(
        max_epochs=cfg.trainer.epochs,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        logger=wandb_logger,
        deterministic=cfg.trainer.deterministic,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
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

    # Get the channel matrix
    if cfg.communication.awareness == 'aware':
        # Channel Initialization
        ch_matrix = channel_matrix
        snr = cfg.communication.snr
        case = 'Neural Semantic Precoding/Decoding'
    elif cfg.communication.awareness == 'unaware':
        ch_matrix = torch.eye(
            cfg.communication.antennas_receiver,
            cfg.communication.antennas_transmitter,
            dtype=torch.complex64,
        )
        snr = None
        case = 'Neural Semantic Precoding/Decoding - Channel Unaware'
    else:
        raise Exception(
            f'Wrong awareness passed: {cfg.communication.awareness}'
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
        grouping=cfg.datamodule.grouping,
        num_workers=cfg.datamodule.workers,
        seed=cfg.seed,
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
    #                 Define the Neural Model
    # =============================================================
    model = NeuralModel(
        datamodule.input_size,
        datamodule.output_size,
        antennas_transmitter=cfg.communication.antennas_transmitter,
        antennas_receiver=cfg.communication.antennas_receiver,
        enc_hidden_dim=datamodule.input_size,
        dec_hidden_dim=datamodule.output_size,
        hidden_size=cfg.model.layers,
        channel_matrix=ch_matrix,
        lmb=cfg.model.lmb,
        snr=snr,
        lr=cfg.model.lr,
    )

    # Training
    trainer.fit(model, datamodule=datamodule)

    model.hparams['channel_matrix'] = channel_matrix
    model.hparams['snr'] = cfg.communication.snr

    nonzero, tot = count_nonzero_weights(model)
    sparsity = 1 - nonzero / tot

    # =============================================================
    #                 Evaluate over the test set
    # =============================================================
    # Get the z_psi_hat
    z_psi_hat = torch.cat(
        trainer.predict(model=model, datamodule=datamodule, ckpt_path='best')
    )

    # Alignment loss
    alignment_metrics = trainer.test(
        model=model, datamodule=datamodule, ckpt_path='best'
    )[0]

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
            'Grouping': cfg.datamodule.grouping,
            'Method': cfg.datamodule.method,
            'Seed': cfg.seed,
            'Antennas Transmitter': cfg.communication.antennas_transmitter,
            'Antennas Receiver': cfg.communication.antennas_receiver,
            'Sparsity': sparsity,
            'Lambda': cfg.model.lmb,
            'SNR': cfg.communication.snr,
            'Awareness': cfg.communication.awareness,
            'Cost': cfg.transmitter.px_cost,
            'FLOPs': neural_sparse_flops(
                transmitter=cfg.communication.antennas_transmitter,
                receiver=cfg.communication.antennas_receiver,
                sparsity=sparsity,
                input_dim=datamodule.input_size,
                output_dim=datamodule.output_size,
                enc_hidden=datamodule.input_size,
                dec_hidden=datamodule.output_size,
                hidden_size=cfg.model.layers,
            ),
            'Accuracy': clf_metrics['test/acc_epoch'],
            'Alignment Loss': alignment_metrics['test/loss_epoch'],
            'Classifier Loss': clf_metrics['test/loss_epoch'],
            'Receiver Model': cfg.receiver.model,
            'Transmitter Model': cfg.transmitter.model,
            'Case': case,
            'Latent Real Dim': datamodule.input_size,
            'Latent Complex Dim': (datamodule.input_size + 1) // 2,
            'Simulation': cfg.simulation,
        }
    ).write_parquet(
        RESULTS_PATH
        / f'{cfg.seed}_{cfg.communication.antennas_transmitter}_{cfg.communication.antennas_receiver}_{cfg.communication.snr}_{cfg.datamodule.dataset}_{cfg.model.lmb}_{cfg.communication.awareness}_{cfg.datamodule.train_label_size}_{cfg.datamodule.grouping}_{cfg.datamodule.method}_{cfg.simulation}.parquet'
    )

    # Closing W&B
    wandb.finish()

    # Cleaning the working space
    remove_non_empty_dir('./wandb/')
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.wandb.project)

    return None


if __name__ == '__main__':
    main()
