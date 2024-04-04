
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

    parser.add_argument('-p',
                        '--path',
                        help="The path towards a dataset.",
                        type=str,
                        required=True)

    parser.add_argument('-f',
                        '--function',
                        help="Activation function to set the right co-domain.",
                        type=str,
                        required=True)

    parser.add_argument('-c',
                        '--columns',
                        help="The column from which the input is retrieved. Default ['x_i', 'a_j']\nNote: You have to pass the columns as an unique str with the following structure 'c1 c2 c3' , this will be parsed as ['c1', 'c2', 'c3'].",
                        default='x_i a_j',
                        type=str)

    parser.add_argument('-t',
                        '--target',
                        help="The target. Default 'r_ij'.",
                        default='r_ij',
                        type=str)

    parser.add_argument('-d',
                        '--dimension',
                        help="The hidden layer dimension. Default 10.",
                        default=10,
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

    args = parser.parse_args()

    columns = args.columns.split()

    datamodule = DataModule(path=args.path, columns=columns, target=args.target, num_workers=args.workers)

    datamodule.prepare_data()
    datamodule.setup()

    model = MultiLayerPerceptron(datamodule.input_size, datamodule.output_size, hidden_dim=args.dimension, hidden_size=args.layers, activ_type=args.function)

    wandb.login()

    wandb_logger = WandbLogger(project=f'SemCom_{args.function}',
                               log_model='all')
    
    trainer = Trainer(logger=wandb_logger,
                      max_epochs=args.epochs)
    trainer.fit(model, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    main()
