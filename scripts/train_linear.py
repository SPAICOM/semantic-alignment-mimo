"""
This python module handles the training of the linear optimizer.

To check available parameters run 'python /path/to/train_linear.py --help'.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

from src.linear_optim import  LinearOptimizer
from src.datamodules import DataModule

def main() -> None:
    """The main loop.
    """
    import argparse

    description = """
    This python module handles the training of the linear optimizer.

    To check available parameters run 'python /path/to/train_linear.py --help'.
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
                        help="The number of anchors.",
                        type=int,
                        required=True)

    parser.add_argument('-s',
                        '--solver',
                        help="The solver for the linear optimizer. Default 'ortho'.",
                        type=str,
                        default='ortho')

    parser.add_argument('--seed',
                        help="The seed for the analysis. Default 42.",
                        default=42,
                        type=int)

    args = parser.parse_args()

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

    # Get the linear optimizer
    opt = LinearOptimizer(solver=args.solver)

    # Fit the optimizer
    opt.fit(datamodule.train_data.z, datamodule.train_data.r_decoder)
    
    # Examples of evaluation
    print(opt.eval(datamodule.train_data.z, datamodule.train_data.r_decoder))
    print(opt.eval(datamodule.val_data.z, datamodule.val_data.r_decoder))
    print(opt.eval(datamodule.test_data.z, datamodule.test_data.r_decoder))
    
    return None


if __name__ == "__main__":
    main()
