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

from src.utils import complex_gaussian_matrix, complex_tensor
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

    parser.add_argument('--method',
                        help="Method for solving the constraint problem. Default 'closed'.",
                        default='closed',
                        type=str)

    parser.add_argument('--mu',
                        help="The mu parameter for admm. Default 1e-4.",
                        default=1e-4,
                        type=float)

    parser.add_argument('--rho',
                        help="The rho parameter for admm. Default 1e3.",
                        default=1e3,
                        type=float)

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
    white_noise_cov = (args.sigma/2) * torch.view_as_complex(torch.stack((torch.eye(args.receiver), torch.eye(args.receiver)), dim=-1))
    
    
    # =========================================================
    #                     Get the dataset
    # =========================================================
    # Initialize the datamodule
    datamodule = DataModule(dataset=args.dataset,
                            encoder=args.encoder,
                            decoder=args.decoder,
                            num_anchors=args.anchors,
                            case='abs',
                            target='abs')
    
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
                             cost=args.cost,
                             mu=args.mu,
                             rho=args.rho)

    # Fit the linear optimizer
    losses, traces = opt.fit(input=datamodule.train_data.z[:1000],
                             output=datamodule.train_data.z_decoder[:1000],
                             iterations=args.iterations,
                             method=args.method)
    
    # Eval the linear optimizer
    print("loss:",
          opt.eval(input=datamodule.test_data.z,
                   output=datamodule.test_data.z_decoder))

    input = complex_tensor(datamodule.test_data.z.T)
    
    print("trace(F^H F):", torch.trace(opt.F.H@opt.F).real.item())
    if args.cost:
        print(torch.trace(opt.F.H@opt.F).real.item() <= args.cost)

    # print("lambda:", opt.lmb)
    import polars as pl
    pl.DataFrame({'Iterations':range(0, len(losses)),
                  'Losses': losses,
                  'Traces': traces}).write_parquet('convergence.parquet')

    print(pl.read_parquet('convergence.parquet'))
    
    fig, axs = plt.subplots(ncols=2, nrows=2)
    plot = sns.lineplot(x=range(0, len(losses)), y=losses, ax=axs[0, 0]).set(title="Convergence", ylabel="MSE Loss", xlabel="Iteration")
    plot = sns.lineplot(x=range(0, len(losses)), y=losses, ax=axs[0, 1]).set(title="Convergence Log Scaled", ylabel="MSE Loss", xlabel="Iteration", yscale='log')
    plot = sns.lineplot(x=range(0, len(traces)), y=traces, ax=axs[1, 0]).set(title="Convergence", ylabel="tr F^HF", xlabel="Iteration")
    plot = sns.lineplot(x=range(0, len(traces)), y=traces, ax=axs[1, 1]).set(title="Convergence Log Scaled", ylabel="tr F^HF", xlabel="Iteration", yscale='log')
    plt.show()

    return None

if __name__ == "__main__":
    main()
