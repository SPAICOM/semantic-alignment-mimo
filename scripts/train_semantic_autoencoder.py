"""
This python module handles the training of the Semantic AutoEncoders.

To check available parameters run 'python /path/to/train_semantic_autoencoder.py --help'.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, BatchSizeFinder, ModelPruning

from src.models import SemanticAutoEncoder
from src.utils import complex_gaussian_matrix, complex_tensor
from src.datamodules import DataModule    

def compute_amount(epoch) -> float:
    # the sum of all returned values need to be smaller than 1
    if epoch == 10:
        return 0.5

    elif epoch == 50:
        return 0.25

    elif 75 < epoch < 99:
        return 0.01


def main() -> None:
    """The main script loop.
    """
    import argparse

    description = """
    This python module handles the training of the Semantic AutoEncoder.

    To check available parameters run 'python /path/to/train_semantic_autoencoder.py --help'.
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-d',
                        '--dataset',
                        help="The dataset.",
                        type=str,
                        required=True)

    parser.add_argument('--encoder',
                        help="The encoder.",
                        type=str,
                        required=True)

    parser.add_argument('--decoder',
                        help="The encoder.",
                        type=str,
                        required=True)

    parser.add_argument('--transmitter',
                        help="The number of antennas for the transmitter.",
                        type=int,
                        required=True)
    
    parser.add_argument('--receiver',
                        help="The number of antennas for the receiver.",
                        type=int,
                        required=True)

    parser.add_argument('--aware',
                        help="The aweraness of the model. Default True.",
                        default=True,
                        type=bool,
                        action=argparse.BooleanOptionalAction)

    parser.add_argument('--prune',
                        help="The aweraness of the model. Default True.",
                        default=False,
                        type=bool,
                        action=argparse.BooleanOptionalAction)

    parser.add_argument('--snr',
                        help="The snr of the communication channel in dB. Set to None if unaware. Default None.",
                        default=None,
                        type=float)

    parser.add_argument('--encneurons',
                        help="The encoder hidden layer dimension.",
                        default=192,
                        type=int)

    parser.add_argument('--decneurons',
                        help="The dec hidden layer dimensionsion.",
                        default=384,
                        type=int)

    parser.add_argument('-l',
                        '--layers',
                        help="The number of the hidden layers. Default 10.",
                        default=10,
                        type=int)

    parser.add_argument('-w',
                        '--workers',
                        help="Number of workers. Default 0.",
                        default=0,
                        type=int)

    parser.add_argument('-e',
                        '--epochs',
                        help="The maximum number of epochs. Default -1.",
                        default=-1,
                        type=int)

    parser.add_argument('-m',
                        '--mu',
                        help="The mu coefficient for the regularizer. Default 1.",
                        default=1.,
                        type=float)

    parser.add_argument('--cost',
                        help="Transmission cost. Default None.",
                        default=None,
                        type=int)

    parser.add_argument('--lr',
                        help="The learning rate. Default 1e-3.",
                        default=1e-3,
                        type=float)

    parser.add_argument('--seed',
                        help="The seed for the analysis. Default 42.",
                        default=42,
                        type=int)

    args = parser.parse_args()

    # Setting the seed
    seed_everything(args.seed, workers=True)

    # Get the channel matrix
    if args.aware:
        aware = 'aware'
        channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(args.receiver, args.transmitter))
        snr = args.snr
    else:
        aware = 'unaware'
        channel_matrix = torch.eye(args.receiver, args.transmitter, dtype=torch.complex64)
        snr = None

    # Initialize the datamodule
    datamodule = DataModule(dataset=args.dataset,
                            encoder=args.encoder,
                            decoder=args.decoder,
                            num_workers=args.workers)

    # Prepare and setup the data
    datamodule.prepare_data()
    datamodule.setup()
    
    # Initialize the model
    model = SemanticAutoEncoder(datamodule.input_size,
                                datamodule.output_size,
                                antennas_transmitter=args.transmitter,
                                antennas_receiver=args.receiver,
                                enc_hidden_dim=args.encneurons,
                                dec_hidden_dim=args.decneurons,
                                hidden_size=args.layers,
                                channel_matrix=channel_matrix,
                                mu=args.mu,
                                snr=snr,
                                cost=args.cost,
                                lr=args.lr)

    # Callbacks definition
    callbacks = [
        LearningRateMonitor(logging_interval='step',
                            log_momentum=True),
        EarlyStopping(monitor='valid/loss_epoch', patience=10),
        ModelCheckpoint(monitor='valid/loss_epoch',
                        save_top_k=1,
                        mode='min'),
        BatchSizeFinder(mode='binsearch',
                        max_trials=8),
    ]

    # Add pruninig to the callbacks if prune is True
    if args.prune:
        callbacks.append(ModelPruning(pruning_fn='l1_unstructured',
                                      amount=compute_amount,
                                      make_pruning_permanent=True,
                                      use_lottery_ticket_hypothesis=True,
                                      resample_parameters=True,
                                      use_global_unstructured=True))
    
    # W&B login and Logger intialization
    wandb.login()
    wandb_logger = WandbLogger(project=f'SemanticAutoEncoder_wn_{args.transmitter}_{args.receiver}_{aware}_{snr}_{args.cost}',
                               name=f"seed_{args.seed}",
                               id=f"seed_{args.seed}",
                               log_model='all')
    
    trainer = Trainer(num_sanity_val_steps=2,
                      max_epochs=args.epochs,
                      logger=wandb_logger,
                      deterministic=True,
                      callbacks=callbacks,
                      log_every_n_steps=10)

    # Training
    trainer.fit(model, datamodule=datamodule)

    # Testing
    trainer.test(datamodule=datamodule, ckpt_path='best')

    # Closing W&B
    wandb.finish()

    return None


if __name__ == "__main__":
    main()
