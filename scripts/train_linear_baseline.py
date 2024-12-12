"""
This python module handles the training of the linear baseline optimizer.

To check available parameters run 'python /path/to/train_linear_baseline.py --help'.
"""
# Add root to the path
import sys
import torch
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

from pytorch_lightning import Trainer, seed_everything

from src.utils import complex_gaussian_matrix, complex_tensor
from src.linear_optim import  LinearOptimizerBaseline
from src.datamodules import DataModule

def main():
    """The main loop.
    """
    import argparse

    description = """
    This python module handles the training of the linear baseline optimizer.

    To check available parameters run 'python /path/to/train_linear_baseline.py --help'.
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
                        '--sigma',
                        help="The sigma squared of the white noise. Default 0.",
                        type=float,
                        default=1.)
    
    parser.add_argument('--transmitter',
                        help="The number of antennas for the transmitter.",
                        type=int,
                        required=True)
    
    parser.add_argument('--typology',
                        help="The typology of baseline, possible vales 'pre' or 'post'. Default 'pre'.",
                        type=str,
                        default="pre")
    
    parser.add_argument('--receiver',
                        help="The number of antennas for the receiver.",
                        type=int,
                        required=True)

    parser.add_argument('--seed',
                        help="The seed for the analysis. Default 42.",
                        default=42,
                        type=int)

    args = parser.parse_args()

    # Setting the seed
    seed_everything(args.seed, workers=True)

    # Get the channel matrix
    channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(args.receiver, args.transmitter))

    
    # =========================================================
    #                     Get the dataset
    # =========================================================
    # Initialize the datamodule
    datamodule = DataModule(dataset=args.dataset,
                            encoder=args.encoder,
                            decoder=args.decoder)
    
    # Prepare and setup the data
    datamodule.prepare_data()
    datamodule.setup()
    
    
    # =========================================================
    #               Define the Linear Optimizer
    # =========================================================
    opt = LinearOptimizerBaseline(input_dim=datamodule.input_size,
                                  output_dim=datamodule.output_size,
                                  channel_matrix=channel_matrix,
                                  sigma=args.sigma,
                                  typology=args.typology)

    # Fit the linear optimizer
    opt.fit(input=datamodule.train_data.z,
            output=datamodule.train_data.z_decoder)
    
    # Eval the linear optimizer
    print("loss:",
          opt.eval(input=datamodule.test_data.z,
                   output=datamodule.test_data.z_decoder))

    return None

if __name__ == "__main__":
    main()
