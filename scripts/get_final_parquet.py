
"""Script for getting the final parquet with the results.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import torch
import polars as pl
from timm import create_model
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import TensorDataset, DataLoader

from src.linear_optim import LinearOptimizerRE, LinearOptimizerSAE
from src.utils import complex_gaussian_matrix, complex_tensor
from src.models import RelativeEncoder, Classifier, SemanticAutoEncoder
from src.datamodules import DataModule, DataModuleClassifier

def main() -> None:
    """The main loop. 
    """
    # Defining paths
    CURRENT: Path = Path('.')
    DATA_DIR: Path = CURRENT / "data"
    MODELS_DIR: Path = CURRENT / "models"

    batch_size = 512
    solvers = ['ortho', 'free']
    trainer = Trainer(inference_mode=True, enable_progress_bar=False, logger=False)
    
    results = pl.DataFrame()
    # ==============================================================================
    #                      Get results for the SAE absolute
    # ==============================================================================
    target = 'abs'
    for autoencoder_path in (MODELS_DIR / f'autoencoders/{target}').rglob('*.ckpt'):
        break
        # Getting the settings
        _, _, _, dataset, encoder, decoder, case, awareness, antennas, seed = str(autoencoder_path.as_posix()).split('/')
        transmitter, receiver = list(map(int, antennas.split('_')[-2:]))
        case = case.split('_')[-1]
        seed = int(seed.split('.')[0].split('_')[-1])

        # Setting the seed
        seed_everything(seed, workers=True)

        # Get the channel matrix
        channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(receiver, transmitter))

        # Get and setup the datamodule
        datamodule = DataModule(dataset=dataset,
                                encoder=encoder,
                                decoder=decoder,
                                num_anchors=None,
                                case=case,
                                target=target,
                                batch_size=batch_size)
        datamodule.prepare_data()
        datamodule.setup()
                
        # =========================================================================
        #                            Classfier Stuff
        # =========================================================================
        # Define the path towards the classifier
        clf_path: Path = MODELS_DIR / f"classifiers/{target}/{dataset}/{decoder}/seed_{seed}.ckpt"

        # Load the classifier model
        clf = Classifier.load_from_checkpoint(clf_path)
        clf.eval()

        # Get and setup the classifier datamodule
        clf_datamodule = DataModuleClassifier(dataset=dataset,
                                              decoder=decoder,
                                              num_anchors=None,
                                              case=target,
                                              batch_size=batch_size)
        clf_datamodule.prepare_data()
        clf_datamodule.setup()

        # =========================================================================
        #                             Non Linear
        # =========================================================================
        model = SemanticAutoEncoder.load_from_checkpoint(autoencoder_path)
        model.eval()

        if awareness == 'unaware':
            model.hparams['channel_matrix'] = channel_matrix
        
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
                           'Case': f'Channel {awareness} {target} SAE NN',
                           'Transmitting Antennas': transmitter,
                           'Receiving Antennas': receiver,
                           'Awareness': awareness,
                           'Target': target,
                           'Anchors': None,
                           'Seed': seed,
                           'Alignment Loss': alignment_metrics['test/loss_epoch'],
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch']
                       }),
                       in_place=True)


        # =========================================================================
        #                             Linear Optimizer
        # =========================================================================
        # Set awareness
        if awareness == 'aware':
            ch_matrix = channel_matrix
        elif awareness == 'unaware':
            ch_matrix = complex_tensor(torch.eye(receiver, transmitter))
        else:
            raise Exception(f'Wrong awareness passed: {awareness}')
                
        # Get the optimizer
        opt = LinearOptimizerSAE(input_dim=datamodule.input_size,
                                 output_dim=datamodule.output_size,
                                 channel_matrix=ch_matrix)

        # Fit the linear optimizer
        opt.fit(input=datamodule.train_data.z,
                output=datamodule.train_data.z_decoder)

        # Set the channel matrix
        opt.channel_matrix = channel_matrix

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
                           'Case': f'Channel {awareness} {target} SAE linear',
                           'Transmitting Antennas': transmitter,
                           'Receiving Antennas': receiver,
                           'Awareness': awareness,
                           'Target': target,
                           'Anchors': None,
                           'Seed': seed,
                           'Alignment Loss': opt.eval(datamodule.test_data.z, datamodule.test_data.z_decoder),
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch']
                       }),
                       in_place=True)

    results = pl.read_parquet('final_results.parquet')
    results = results.with_columns(pl.col('Anchors').cast(pl.Int64))
    print(results)
    results.write_parquet('final_results.parquet')

    # ==============================================================================
    #                      Get results for the SAE relative
    # ==============================================================================
    target = 'rel'
    for autoencoder_path in (MODELS_DIR / f'autoencoders/{target}').rglob('*.ckpt'):
        # Getting the settings
        _, _, _, dataset, encoder, decoder, case, awareness, antennas, anchors, seed = str(autoencoder_path.as_posix()).split('/')
        transmitter, receiver = list(map(int, antennas.split('_')[-2:]))
        case = case.split('_')[-1]
        anchors = int(anchors.split('_')[-1])
        seed = int(seed.split('.')[0].split('_')[-1])

        # Setting the seed
        seed_everything(seed, workers=True)

        # Get the channel matrix
        channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(receiver, transmitter))

        # Get and setup the datamodule
        datamodule = DataModule(dataset=dataset,
                                encoder=encoder,
                                decoder=decoder,
                                num_anchors=anchors,
                                case=case,
                                target=target,
                                batch_size=batch_size)
        datamodule.prepare_data()
        datamodule.setup()
                
        # =========================================================================
        #                            Classfier Stuff
        # =========================================================================
        # Define the path towards the classifier
        clf_path: Path = MODELS_DIR / f"classifiers/{target}/{dataset}/{decoder}/anchors_{anchors}/seed_{seed}.ckpt"

        # Load the classifier model
        clf = Classifier.load_from_checkpoint(clf_path)
        clf.eval()

        # Get and setup the classifier datamodule
        clf_datamodule = DataModuleClassifier(dataset=dataset,
                                              decoder=decoder,
                                              num_anchors=anchors,
                                              case=target,
                                              batch_size=batch_size)
        clf_datamodule.prepare_data()
        clf_datamodule.setup()

        # =========================================================================
        #                             Non Linear
        # =========================================================================
        model = SemanticAutoEncoder.load_from_checkpoint(autoencoder_path)
        model.eval()
        
        if awareness == 'unaware':
            model.hparams['channel_matrix'] = channel_matrix
        
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
                           'Case': f'Channel {awareness} {target} SAE NN',
                           'Transmitting Antennas': transmitter,
                           'Receiving Antennas': receiver,
                           'Awareness': awareness,
                           'Target': target,
                           'Anchors': anchors,
                           'Seed': seed,
                           'Alignment Loss': alignment_metrics['test/loss_epoch'],
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch']
                       }),
                       in_place=True)


        # =========================================================================
        #                             Linear Optimizer
        # =========================================================================
        # Set awareness
        if awareness == 'aware':
            ch_matrix = channel_matrix
        elif awareness == 'unaware':
            ch_matrix = complex_tensor(torch.eye(receiver, transmitter))
        else:
            raise Exception(f'Wrong awareness passed: {awareness}')
                
        # Get the optimizer
        opt = LinearOptimizerSAE(input_dim=datamodule.input_size,
                                 output_dim=datamodule.output_size,
                                 channel_matrix=ch_matrix)

        # Fit the linear optimizer
        opt.fit(input=datamodule.train_data.z,
                output=datamodule.train_data.r_decoder)

        # Set the channel matrix
        opt.channel_matrix = channel_matrix

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
                           'Case': f'Channel {awareness} {target} SAE linear',
                           'Transmitting Antennas': transmitter,
                           'Receiving Antennas': receiver,
                           'Awareness': awareness,
                           'Target': target,
                           'Anchors': anchors,
                           'Seed': seed,
                           'Alignment Loss': opt.eval(datamodule.test_data.z, datamodule.test_data.r_decoder),
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch']
                       }),
                       in_place=True)
                
    print(results)
    results.write_parquet('final_results.parquet')

    return None

    

if __name__ == "__main__":
    main()
