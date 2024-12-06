
"""Script for getting the final parquet with the results.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import math
import torch
import polars as pl
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import TensorDataset, DataLoader

from src.linear_optim import LinearOptimizerSAE, LinearOptimizerBaseline
from src.utils import complex_gaussian_matrix, complex_tensor, snr
from src.models import Classifier, SemanticAutoEncoder
from src.datamodules import DataModule, DataModuleClassifier

def main() -> None:
    """The main loop. 
    """
    import argparse

    description = """
    This python module handles the parquet file generation.

    To check available parameters run 'python /path/to/get_final_parquet.py --help'.
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c',
                        '--checkpoints',
                        help="The checkpoints path. Default 'autoencoders'.",
                        default='autoencoders',
                        type=str)

    parser.add_argument('-n',
                        '--name',
                        help="The parquet file name. Default 'final_results'.",
                        default='final_results',
                        type=str)

    args = parser.parse_args()
    
    # Defining paths
    CURRENT: Path = Path('.')
    DATA_DIR: Path = CURRENT / "data"
    MODELS_DIR: Path = CURRENT / "models"

    batch_size = 512
    trainer = Trainer(inference_mode=True, enable_progress_bar=False, logger=False)
    
    try:
        results = pl.read_parquet(f'{args.name}.parquet')
    except:
        results = pl.DataFrame(schema=[
                                   ('Dataset', pl.String),
                                   ('Encoder', pl.String),
                                   ('Decoder', pl.String),
                                   ('Case', pl.String),
                                   ('Transmitting Antennas', pl.Int64),
                                   ('Receiving Antennas', pl.Int64),
                                   ('Awareness', pl.String),
                                   ('Sigma', pl.Float64),
                                   ('Seed', pl.Int64),
                                   ('Cost', pl.Int64),
                                   ('Alignment Loss', pl.Float64),
                                   ('Classifier Loss', pl.Float64),
                                   ('Accuracy', pl.Float64),
                                   ('SNR', pl.Float64)
                               ])
    # ==============================================================================
    #                      Get results for the SAE absolute
    # ==============================================================================
    for autoencoder_path in (MODELS_DIR / f'{args.checkpoints}/').rglob('*.ckpt'):
        print()
        print()
        print('#'*100)
        print(autoencoder_path)
        # Getting the settings
        _, _, dataset, encoder, decoder, awareness, antennas, sigma, seed = str(autoencoder_path.as_posix()).split('/')
        transmitter, receiver = list(map(int, antennas.split('_')[-2:]))
        # case = case.split('_')[-1]
        sigma = float(sigma.split('_')[-1])
        c_sigma = sigma / math.sqrt(2)
        seed = int(seed.split('.')[0].split('_')[-1])

        if not results.filter((pl.col('Transmitting Antennas')==transmitter)&(pl.col('Receiving Antennas')==receiver)&(pl.col('Seed')==seed)&(pl.col('Sigma')==sigma)&(pl.col('Awareness')==awareness)).is_empty():
            continue
        
        # Setting the seed
        seed_everything(seed, workers=True)

        # Get the channel matrix and white noise sigma
        channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(receiver, transmitter))

        # Get and setup the datamodule
        datamodule = DataModule(dataset=dataset,
                                encoder=encoder,
                                decoder=decoder,
                                batch_size=batch_size)
        datamodule.prepare_data()
        datamodule.setup()
                
        # =========================================================================
        #                            Classfier Stuff
        # =========================================================================
        # Define the path towards the classifier
        clf_path: Path = MODELS_DIR / f"classifiers/{dataset}/{decoder}/seed_{seed}.ckpt"

        # Load the classifier model
        clf = Classifier.load_from_checkpoint(clf_path)
        clf.eval()

        # Get and setup the classifier datamodule
        clf_datamodule = DataModuleClassifier(dataset=dataset,
                                              decoder=decoder,
                                              batch_size=batch_size)
        clf_datamodule.prepare_data()
        clf_datamodule.setup()

        # =========================================================================
        #                             Non Linear
        # =========================================================================
        model = SemanticAutoEncoder.load_from_checkpoint(autoencoder_path)
        model.eval()

        cost = model.hparams['cost']

        if awareness == 'unaware':
            model.hparams['channel_matrix'] = channel_matrix
            model.hparams['sigma'] = sigma
        
        # Get the absolute representation in the decoder space
        z_psi_hat = torch.cat(trainer.predict(model=model, datamodule=datamodule))
        
        alignment_metrics = trainer.test(model=model, datamodule=datamodule)[0]
      
        # Get the predictions using as input the z_psi_hat
        dataloader = DataLoader(TensorDataset(z_psi_hat, datamodule.test_data.labels), batch_size=batch_size)
        clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]

        results.vstack(pl.DataFrame(
                       {
                           'Dataset': dataset,
                           'Encoder': encoder,
                           'Decoder': decoder,
                           'Case': f'Neural Semantic Precoding/Decoding - Channel {awareness.capitalize()}',
                           'Transmitting Antennas': transmitter,
                           'Receiving Antennas': receiver,
                           'Awareness': awareness,
                           'Sigma': sigma,
                           'Seed': seed,
                           'Cost': cost,
                           'Alignment Loss': alignment_metrics['test/loss_epoch'],
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch'],
                           'SNR': snr(signal=model.get_precodings(datamodule.test_data.z).H, sigma=c_sigma)
                       }),
                       in_place=True)


        # =========================================================================
        #                           Linear Optimizer SAE
        # =========================================================================
        # Set awareness
        if awareness == 'aware':
            ch_matrix = channel_matrix
            sigma_0 = sigma
        elif awareness == 'unaware':
            ch_matrix = torch.view_as_complex(torch.stack((torch.eye(receiver), torch.eye(receiver)), dim=-1))
            sigma_0 = 0
        else:
            raise Exception(f'Wrong awareness passed: {awareness}')
                
        # Get the optimizer
        opt = LinearOptimizerSAE(input_dim=datamodule.input_size,
                                 output_dim=datamodule.output_size,
                                 channel_matrix=ch_matrix,
                                 sigma=sigma_0,
                                 cost=cost)

        # Fit the linear optimizer
        opt.fit(input=datamodule.train_data.dataset.z[:1000],
                output=datamodule.train_data.dataset.z_decoder[:1000],
                iterations=30)

        # Set the channel matrix and white noise sigma
        opt.channel_matrix = channel_matrix
        opt.sigma = sigma

        # Get the z_psi_hat
        z_psi_hat = opt.transform(datamodule.test_data.z)

        # Get the predictions using as input the z_psi_hat
        dataloader = DataLoader(TensorDataset(z_psi_hat, datamodule.test_data.labels), batch_size=batch_size)
        clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]

        results.vstack(pl.DataFrame(
                       {
                           'Dataset': dataset,
                           'Encoder': encoder,
                           'Decoder': decoder,
                           'Case': f'Linear Semantic Precoding/Decoding - Channel {awareness.capitalize()}',
                           'Transmitting Antennas': transmitter,
                           'Receiving Antennas': receiver,
                           'Awareness': awareness,
                           'Sigma': sigma,
                           'Seed': seed,
                           'Cost': cost,
                           'Alignment Loss': opt.eval(datamodule.test_data.z, datamodule.test_data.z_decoder),
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch'],
                           'SNR': snr(signal=opt.get_precodings(datamodule.test_data.z), sigma=c_sigma)
                       }),
                       in_place=True)

        
        # =========================================================================
        #                        Linear Optimizer Baseline
        # =========================================================================
        # Set awareness
        if awareness == 'aware':
            ch_matrix = channel_matrix
            sigma_0 = sigma
        elif awareness == 'unaware':
            ch_matrix = torch.view_as_complex(torch.stack((torch.eye(receiver), torch.eye(receiver)), dim=-1))
            sigma_0 = 0
        else:
            raise Exception(f'Wrong awareness passed: {awareness}')
                
        # Get the optimizer
        opt = LinearOptimizerBaseline(input_dim=datamodule.input_size,
                                      output_dim=datamodule.output_size,
                                      channel_matrix=ch_matrix,
                                      sigma=sigma_0)

        # Fit the linear optimizer
        opt.fit(input=datamodule.train_data.dataset.z[:1000],
                output=datamodule.train_data.dataset.z_decoder[:1000])

        # Set the channel matrix and white noise sigma
        opt.channel_matrix = channel_matrix
        opt.sigma = sigma

        # Get the z_psi_hat
        z_psi_hat = opt.transform(datamodule.test_data.z)

        # Get the predictions using as input the z_psi_hat
        dataloader = DataLoader(TensorDataset(z_psi_hat, datamodule.test_data.labels), batch_size=batch_size)
        clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]

        results.vstack(pl.DataFrame(
                       {
                           'Dataset': dataset,
                           'Encoder': encoder,
                           'Decoder': decoder,
                           'Case': f'Baseline - Channel {awareness.capitalize()}',
                           'Transmitting Antennas': transmitter,
                           'Receiving Antennas': receiver,
                           'Awareness': awareness,
                           'Sigma': sigma,
                           'Seed': seed,
                           'Cost': cost,
                           'Alignment Loss': opt.eval(datamodule.test_data.z, datamodule.test_data.z_decoder),
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch'],
                           'SNR': snr(signal=torch.ones(1), sigma=c_sigma)
                       }),
                       in_place=True)

        results.write_parquet(f'{args.name}.parquet')
        
    print(results)
    
    return None
    

if __name__ == "__main__":
    main()
