"""
This python module handles the training of our models.

To check available parameters run 'python /path/to/main.py --help'.
"""

import wandb
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.datamodule import DataModule
from src.models import MultiLayerPerceptron

def main() -> None:
    """The main script loop.
    """
    import argparse

    description = """
    This python module handles the training of our models.

    To check available parameters run 'python /path/to/main.py --help'.
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

    parser.add_argument('-s',
                        '--split',
                        help="The split.",
                        type=str,
                        required=True)

    parser.add_argument('-a',
                        '--anchors',
                        help="The number of anchors.",
                        type=int,
                        required=True)

    parser.add_argument('-c',
                        '--case',
                        help="The case of the network input. Default 'abs'.",
                        default='abs',
                        type=str)

    parser.add_argument('-f',
                        '--function',
                        help="Activation function to set the right co-domain.",
                        choices=['tanh', 'sigmoid', 'softplus', 'relu'],
                        type=str,
                        required=True)

    parser.add_argument('-n',
                        '--neurons',
                        help="The hidden layer dimension.",
                        required=True,
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
                        help="The maximum number of epochs. Default 10.",
                        default=10,
                        type=int)

    parser.add_argument('-b',
                        '--batch',
                        help="The batch size. Default 128.",
                        default=128,
                        type=int)

    args = parser.parse_args()

    datamodule = DataModule(dataset=args.dataset,
                            encoder=args.encoder,
                            decoder=args.decoder,
                            split=args.split,
                            num_anchors=args.anchors,
                            case=args.case,
                            num_workers=args.workers,
                            batch_size=args.batch)

    datamodule.prepare_data()
    datamodule.setup()

    model = MultiLayerPerceptron(datamodule.input_size,
                                 datamodule.output_size,
                                 hidden_dim=args.neurons,
                                 hidden_size=args.layers,
                                 activ_type=args.function)

    wandb.login()

    wandb_logger = WandbLogger(project=f'SemCom_{args.function}_{args.case}',
                               log_model='all')
    
    trainer = Trainer(logger=wandb_logger,
                      max_epochs=args.epochs)
    trainer.fit(model, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    main()
