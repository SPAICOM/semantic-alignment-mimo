"""
This python module handles the training of the linear optimizer for SAE.

To check available parameters run 'python /path/to/train_linear_SAE.py --help'.
"""
# Add root to the path
import sys
import torch
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(str(Path(sys.path[0]).parent))

from pytorch_lightning import Trainer, seed_everything

from src.utils import complex_gaussian_matrix
from src.linear_optim import  LinearOptimizerSAE
from src.datamodules import DataModule

def main():
    """The main loop.
    """
    import argparse

    description = """
    This python module handles the training of the linear optimizer for SAE.

    To check available parameters run 'python /path/to/train_linear_SAE.py --help'.
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

    parser.add_argument('-a',
                        '--anchors',
                        help="The number of anchors. Default None.",
                        type=int,
                        default=None)
    
    parser.add_argument('-s',
                        '--sigma',
                        help="The sigma squared of the white noise. Default 0.",
                        type=int,
                        default=0)
    
    parser.add_argument('--transmitter',
                        help="The number of antennas for the transmitter.",
                        type=int,
                        required=True)
    
    parser.add_argument('--receiver',
                        help="The number of antennas for the receiver.",
                        type=int,
                        required=True)

    parser.add_argument('-i',
                        '--iterations',
                        help="The number of fitting iterations. Default None.",
                        type=int,
                        default=None)
    
    parser.add_argument('--cost',
                        help="Transmission cost. Default None.",
                        default=None,
                        type=int)

    parser.add_argument('--seed',
                        help="The seed for the analysis. Default 42.",
                        default=42,
                        type=int)

    args = parser.parse_args()

    # Setting the seed
    seed_everything(args.seed, workers=True)

    # Get the channel matrix
    channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(args.receiver, args.transmitter))

    # Set the white noise
    white_noise_cov = args.sigma * torch.eye(args.receiver)
    
    # =========================================================
    #                     Get the dataset
    # =========================================================
    # Initialize the datamodule
    datamodule = DataModule(dataset=args.dataset,
                            encoder=args.encoder,
                            decoder=args.decoder,
                            num_anchors=args.anchors,
                            case='abs')
    
    # Prepare and setup the data
    datamodule.prepare_data()
    datamodule.setup()

    # =========================================================
    #               Define the Linear Optimizer
    # =========================================================
    opt = LinearOptimizerSAE(input_dim=datamodule.input_size,
                             output_dim=datamodule.output_size,
                             channel_matrix=channel_matrix,
                             white_noise_cov=white_noise_cov,
                             sigma=args.sigma,
                             cost=args.cost)

    # Fit the linear optimizer
    losses = opt.fit(input=datamodule.train_data.z,
                     output=datamodule.train_data.z_decoder,
                     iterations=args.iterations)
    
    # Eval the linear optimizer
    print("loss:",
          opt.eval(input=datamodule.test_data.z,
                   output=datamodule.test_data.z_decoder))

    print("trace(F^H F):", torch.trace(opt.F.H @ opt.F).item())

    # print("lambda:", opt.lmb)
    
    plot = sns.lineplot(x=range(0, len(losses)), y=losses).set(title="Convergence", ylabel="MSE Loss", xlabel="Iteration")#, yscale='log')
    plt.show()

    return None

if __name__ == "__main__":
    main()
