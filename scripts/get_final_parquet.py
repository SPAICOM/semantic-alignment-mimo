
"""Script for getting the final parquet with the results.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import math
import torch
import polars as pl
from tqdm.auto import tqdm
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import TensorDataset, DataLoader

from src.linear_optim import LinearOptimizerSAE, LinearOptimizerBaseline
from src.utils import complex_gaussian_matrix, complex_tensor, sigma_given_snr
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

    parser.add_argument('--pre',
                        help="Perform the pre baseline. Default True.",
                        default=True,
                        type=bool,
                        action=argparse.BooleanOptionalAction)

    parser.add_argument('--post',
                        help="Perform the post baseline. Default True.",
                        default=True,
                        type=bool,
                        action=argparse.BooleanOptionalAction)

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
                                   ('Symbols', pl.Int64),
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
    for autoencoder_path in tqdm(list((MODELS_DIR / f'{args.checkpoints}/').rglob('*.ckpt'))):
        print()
        print()
        print('#'*150)
        print(autoencoder_path)
        # Getting the settings
        _, _, dataset, encoder, decoder, awareness, antennas, snr, seed = str(autoencoder_path.as_posix()).split('/')
        transmitter, receiver = list(map(int, antennas.split('_')[-2:]))
        snr = float(snr.split('_')[-1])
        seed = int(seed.split('.')[0].split('_')[-1])

        if not results.filter((pl.col('Transmitting Antennas')==transmitter)&(pl.col('Receiving Antennas')==receiver)&(pl.col('Seed')==seed)&(pl.col('SNR')==snr)&(pl.col('Awareness')==awareness)).is_empty():
            continue
        
        # Setting the seed
        seed_everything(seed, workers=True)

        # Get the channel matrix
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
        case = 'Neural Semantic Precoding/Decoding'

        if awareness == 'unaware':
            model.hparams['channel_matrix'] = channel_matrix
            model.hparams['snr'] = snr
            case = 'Neural Semantic Precoding/Decoding - Channel Unaware'
        
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
                           'Case': case,
                           'Symbols': transmitter,
                           'Transmitting Antennas': transmitter,
                           'Receiving Antennas': receiver,
                           'Awareness': awareness,
                           'Sigma': sigma_given_snr(snr, model.get_precodings(datamodule.test_data.z).H),
                           'Seed': seed,
                           'Cost': cost,
                           'Alignment Loss': alignment_metrics['test/loss_epoch'],
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch'],
                           'SNR': snr
                       }),
                       in_place=True)


        # =========================================================================
        #                           Linear Optimizer SAE
        # =========================================================================
        # Set awareness
        if awareness == 'aware':
            ch_matrix = channel_matrix
            snr_0 = snr
            case = 'Linear Semantic Precoding/Decoding'
        elif awareness == 'unaware':
            ch_matrix = torch.eye(receiver, transmitter, dtype=torch.complex64)
            snr_0 = None
            case = 'Linear Semantic Precoding/Decoding - Channel Unaware'
        else:
            raise Exception(f'Wrong awareness passed: {awareness}')
                
        # Get the optimizer
        opt = LinearOptimizerSAE(input_dim=datamodule.input_size,
                                 output_dim=datamodule.output_size,
                                 channel_matrix=ch_matrix,
                                 snr=snr_0,
                                 cost=cost)

        # Fit the linear optimizer
        opt.fit(input=datamodule.train_data.z,
                output=datamodule.train_data.z_decoder,
                iterations=20)

        # Set the channel matrix and the snr
        opt.channel_matrix = channel_matrix
        opt.snr = snr

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
                           'Case': case,
                           'Symbols': transmitter,
                           'Transmitting Antennas': transmitter,
                           'Receiving Antennas': receiver,
                           'Awareness': awareness,
                           'Sigma': sigma_given_snr(snr, opt.get_precodings(datamodule.test_data.z)),
                           'Seed': seed,
                           'Cost': cost,
                           'Alignment Loss': opt.eval(datamodule.test_data.z, datamodule.test_data.z_decoder),
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch'],
                           'SNR': snr
                       }),
                       in_place=True)

        
        # =========================================================================
        #               Linear Optimizer Baseline Alignment pre SVD
        # =========================================================================
        if args.pre and awareness != "unaware":
            print("Baseline Pre")
            
            # Get the optimizer
            opt = LinearOptimizerBaseline(input_dim=datamodule.input_size,
                                          output_dim=datamodule.output_size,
                                          channel_matrix=ch_matrix,
                                          snr=snr_0,
                                          typology="pre")

            # Fit the linear optimizer
            opt.fit(input=datamodule.train_data.z,
                    output=datamodule.train_data.z_decoder)

            # Set the channel matrix and the snr
            opt.channel_matrix = channel_matrix
            opt.snr = snr

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
                               'Case': f'Baseline No Semantic Compression',
                               'Symbols': datamodule.test_data.output_size // 2,
                               'Transmitting Antennas': transmitter,
                               'Receiving Antennas': receiver,
                               'Awareness': awareness,
                               'Sigma': sigma_given_snr(snr, opt.get_precodings(datamodule.test_data.z)),
                               'Seed': seed,
                               'Cost': cost,
                               'Alignment Loss': opt.eval(datamodule.test_data.z, datamodule.test_data.z_decoder),
                               'Classifier Loss': clf_metrics['test/loss_epoch'],
                               'Accuracy': clf_metrics['test/acc_epoch'],
                               'SNR': snr
                           }),
                           in_place=True)

        # =========================================================================
        #               Linear Optimizer Baseline Alignment post SVD
        # =========================================================================
        if args.post and awareness != "unaware":
            print("Baseline Post")
            # Get the optimizer
            opt = LinearOptimizerBaseline(input_dim=datamodule.input_size,
                                          output_dim=datamodule.output_size,
                                          channel_matrix=ch_matrix,
                                          snr=snr_0,
                                          typology="post")

            # Fit the linear optimizer
            opt.fit(input=datamodule.train_data.z,
                    output=datamodule.train_data.z_decoder)

            # Set the channel matrix and the snr
            opt.channel_matrix = channel_matrix
            opt.snr = snr

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
                               'Case': f'Baseline No Semantic Compression',
                               'Symbols': datamodule.test_data.input_size // 2,
                               'Transmitting Antennas': transmitter,
                               'Receiving Antennas': receiver,
                               'Awareness': awareness,
                               'Sigma': sigma_given_snr(snr, opt.get_precodings(datamodule.test_data.z)),
                               'Seed': seed,
                               'Cost': cost,
                               'Alignment Loss': opt.eval(datamodule.test_data.z, datamodule.test_data.z_decoder),
                               'Classifier Loss': clf_metrics['test/loss_epoch'],
                               'Accuracy': clf_metrics['test/acc_epoch'],
                               'SNR': snr
                           }),
                           in_place=True)

        results.write_parquet(f'{args.name}.parquet')
        
    print(results)
    
    return None
    

if __name__ == "__main__":
    main()
