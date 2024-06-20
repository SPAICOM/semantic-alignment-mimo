"""
This python module handles the training of the classifier.

To check available parameters run 'python /path/to/train_classifier.py --help'.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, BatchSizeFinder

from src.datamodules import DataModuleClassifier
from src.models import Classifier

def main() -> None:
    """The main script loop.
    """
    import argparse

    description = """
    This python module handles the training of the classifiers.

    To check available parameters run 'python /path/to/train_classifier.py --help'.
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-d',
                        '--dataset',
                        help="The dataset.",
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

    parser.add_argument('-n',
                        '--neurons',
                        help="The hidden layer dimension.",
                        required=True,
                        type=int)

    parser.add_argument('-c',
                        '--case',
                        help="The case of the input. Default 'abs'.",
                        default='abs',
                        type=str)

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

    # Initialize the datamodule
    datamodule = DataModuleClassifier(dataset=args.dataset,
                                      decoder=args.decoder,
                                      num_anchors=args.anchors,
                                      case=args.case,
                                      num_workers=args.workers)

    # Prepare and setup the data
    datamodule.prepare_data()
    datamodule.setup()

    # Initialize the model
    model = Classifier(datamodule.input_size,
                       datamodule.num_classes,
                       hidden_dim=args.neurons,
                       lr=args.lr,
                       max_lr=0.3)

    # Callbacks definition
    callbacks = [
        LearningRateMonitor(logging_interval='step',
                            log_momentum=True),
        EarlyStopping(monitor='valid/loss_epoch', patience=20),
        ModelCheckpoint(monitor='valid/acc_epoch',
                        mode='max'),
        BatchSizeFinder(mode='binsearch',
                        max_trials=8)
    ]
    
    # W&B login and Logger intialization
    wandb.login()
    wandb_logger = WandbLogger(project=f'Classifier_{args.case}_{args.decoder}',
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
