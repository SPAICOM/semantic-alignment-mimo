"""
This python module handles the training of the linear baseline optimizer.

To check available parameters run 'python /path/to/example_training_baseline.py --help'.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

from pytorch_lightning import seed_everything

from src.datamodules import DataModule
from src.utils import complex_gaussian_matrix
from src.linear_models import  LinearOptimizerBaseline

def main():
    """The main loop.
    """
    import argparse

    description = """
    This python module handles the training of the linear baseline optimizer.

    To check available parameters run 'python /path/to/example_training_baseline.py --help'.
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
                        '--snr',
                        help="The snr of the communication channel. Set to None for unaware. Default None.",
                        type=float,
                        default=None)
    
    parser.add_argument('--snr_type',
                        help="The snr type of the communication channel. Default 'transmitted'.",
                        choices=["transmitted", "received"],
                        type=str,
                        default="transmitted")
    
    parser.add_argument('--transmitter',
                        help="The number of antennas for the transmitter.",
                        type=int,
                        required=True)
    
    parser.add_argument('--typology',
                        help="The typology of baseline, possible vales 'pre' or 'post'. Default 'pre'.",
                        type=str,
                        choices=["pre", "post"],
                        default="pre")
    
    parser.add_argument('--strategy',
                        help="The strategy to apply in sending packets. Default 'first'.",
                        type=str,
                        choices=["first", "abs"],
                        default="first")
    
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
                                  snr=args.snr,
                                  snr_type=args.snr_type,
                                  k_p=1,
                                  typology=args.typology,
                                  strategy=args.strategy)

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
