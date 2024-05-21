"""Script for getting the final parquet with the results.
"""
# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import torch
import polars as pl
from timm import create_model
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader

from src.linear_optim import LinearOptimizer
from src.models import RelativeEncoder, Classifier, SemanticAutoEncoder
from src.datamodules import DataModuleRelativeEncoder, DataModuleClassifier

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
    #                      Get results for the Relative Encoder
    # ==============================================================================
    for encoder_path in (MODELS_DIR / 'encoders').rglob('*.ckpt'):
        # Getting the settings
        _, _, dataset, encoder, decoder, function, case, seed, ckpt = str(encoder_path.as_posix()).split("/")
        seed = int(seed.split('_')[-1])
        anchors = int(ckpt.split('.')[0].split('_')[-1])
        
        # =========================================================================
        #                            Encoder Stuff
        # =========================================================================
        # Load the encoder model
        enc_model = RelativeEncoder.load_from_checkpoint(encoder_path)
        enc_model.eval()
    
        # Get and setup the encoder datamodule
        enc_datamodule = DataModuleRelativeEncoder(dataset=dataset,
                                                   encoder=encoder,
                                                   decoder=decoder,
                                                   num_anchors=anchors,
                                                   case=case,
                                                   batch_size=batch_size)
        enc_datamodule.prepare_data()
        enc_datamodule.setup()
                            
        # =========================================================================
        #                            Classfier Stuff
        # =========================================================================
        # Define the path towards the classifier
        clf_path: Path = MODELS_DIR / f"classifiers/{dataset}/{decoder}/seed_{seed}/anchors_{anchors}.ckpt"

        # Load the classifier model
        clf = Classifier.load_from_checkpoint(clf_path)
        clf.eval()

        # Get and setup the classifier datamodule
        clf_datamodule = DataModuleClassifier(dataset=dataset,
                                              decoder=decoder,
                                              num_anchors=anchors,
                                              batch_size=batch_size)
        clf_datamodule.prepare_data()
        clf_datamodule.setup()

        # =========================================================================
        #                           Computing the results
        # =========================================================================
        # Get the relative representation in the decoder space
        r_psi_hat = torch.cat(trainer.predict(model=enc_model, datamodule=enc_datamodule))
        alignment_metrics = trainer.test(model=enc_model, datamodule=enc_datamodule)[0]
              
        # Get the predictions using as input the r_psi_hat
        dataloader = DataLoader(TensorDataset(r_psi_hat, enc_datamodule.test_data.labels), batch_size=batch_size)
        clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]
        
        results.vstack(pl.DataFrame(
                       {
                           'Dataset': dataset,
                           'Encoder': encoder,
                           'Decoder': decoder,
                           'Function': function,
                           'Case': f'{case} RE NN',
                           'Seed': seed,
                           'Anchors': anchors,
                           'Alignment Loss': alignment_metrics['test/loss_epoch'],
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch']
                       }),
                       in_place=True)

        # =========================================================================
        #                           Linear Optimizer
        # =========================================================================
        # Get the linear optimizer results
        if case == "abs":
            for solver in solvers:
                # Get the Linear Optimizer
                opt = LinearOptimizer(solver=solver)

                # Fit the optimizer
                opt.fit(enc_datamodule.train_data.z, enc_datamodule.train_data.r_decoder)

                # Get the relative representation in the decoder space
                r_psi_hat = opt.transform(enc_datamodule.test_data.z)
                
                # Get the predictions using as input the r_psi_hat
                dataloader = DataLoader(TensorDataset(r_psi_hat, enc_datamodule.test_data.labels), batch_size=batch_size)
                clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]

                results.vstack(pl.DataFrame(
                               {
                                   'Dataset': dataset,
                                   'Encoder': encoder,
                                   'Decoder': decoder,
                                   'Function': function,
                                   'Case': f'{case} RE linear {solver}',
                                   'Seed': seed,
                                   'Anchors': anchors,
                                   'Alignment Loss': opt.eval(enc_datamodule.test_data.z, enc_datamodule.test_data.r_decoder),
                                   'Classifier Loss': clf_metrics['test/loss_epoch'],
                                   'Accuracy': clf_metrics['test/acc_epoch']
                               }),
                               in_place=True)

  
    # ==============================================================================
    #                            Get the original values
    # ==============================================================================
    for clf_path in (MODELS_DIR / 'classifiers').rglob('*.ckpt'):
        # Getting the settings
        _, _, dataset, decoder, seed, ckpt = str(clf_path.as_posix()).split("/")
        seed = int(seed.split('_')[-1])
        anchors = int(ckpt.split('.')[0].split('_')[-1])

        # Load the classifier model
        clf = Classifier.load_from_checkpoint(clf_path)
        clf.eval()

        # Get and setup the classifier datamodule
        clf_datamodule = DataModuleClassifier(dataset=dataset,
                                              decoder=decoder,
                                              num_anchors=anchors,
                                              batch_size=batch_size)
        clf_datamodule.prepare_data()
        clf_datamodule.setup()
        
        # Get the accuracy
        metrics = trainer.test(model=clf, datamodule=clf_datamodule)[0]
        
        results.vstack(pl.DataFrame(
                       {
                           'Dataset': dataset,
                           'Encoder': decoder,
                           'Decoder': decoder,
                           'Function': None,
                           'Case': 'Original Model',
                           'Seed': seed,
                           'Anchors': anchors,
                           'Alignment Loss': None,
                           'Classifier Loss': metrics['test/loss_epoch'],
                           'Accuracy': metrics['test/acc_epoch']
                       }),
                       in_place=True)

    
    # ==============================================================================
    #                   Get results for the Semantic AutoEncoder
    # ==============================================================================
    for autoencoder_path in (MODELS_DIR / 'autoencoders').rglob('*.ckpt'):
        # Getting the settings
        _, _, dataset, encoder, decoder, case, seed, ckpt = str(autoencoder_path.as_posix()).split('/')
        seed = int(seed.split('_')[-1])
        anchors = int(ckpt.split('.')[0].split('_')[-1])

        # Load the Semantic AutoEncoder model
        model = SemanticAutoEncoder.load_from_checkpoint(autoencoder_path)
        model.eval()

        # Get and setup the datamodule
        datamodule = DataModuleRelativeEncoder(dataset=dataset,
                                               encoder=encoder,
                                               decoder=decoder,
                                               num_anchors=anchors,
                                               case=case,
                                               batch_size=batch_size)
        datamodule.prepare_data()
        datamodule.setup()
        
        # =========================================================================
        #                            Classfier Stuff
        # =========================================================================
        # Define the path towards the classifier
        clf_path: Path = MODELS_DIR / f"classifiers/{dataset}/{decoder}/seed_{seed}/anchors_{anchors}.ckpt"

        # Load the classifier model
        clf = Classifier.load_from_checkpoint(clf_path)
        clf.eval()

        # Get and setup the classifier datamodule
        clf_datamodule = DataModuleClassifier(dataset=dataset,
                                              decoder=decoder,
                                              num_anchors=anchors,
                                              batch_size=batch_size)
        clf_datamodule.prepare_data()
        clf_datamodule.setup()

        # =========================================================================
        #                           Computing the results
        # =========================================================================
        # Get the relative representation in the decoder space
        r_psi_hat = torch.cat(trainer.predict(model=model, datamodule=datamodule))
        alignment_metrics = trainer.test(model=model, datamodule=datamodule)[0]
              
        # Get the predictions using as input the r_psi_hat
        dataloader = DataLoader(TensorDataset(r_psi_hat, datamodule.test_data.labels), batch_size=batch_size)
        clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]
        
        results.vstack(pl.DataFrame(
                       {
                           'Dataset': dataset,
                           'Encoder': encoder,
                           'Decoder': decoder,
                           'Function': function,
                           'Case': f'{case} SAE NN',
                           'Seed': seed,
                           'Anchors': anchors,
                           'Alignment Loss': alignment_metrics['test/loss_epoch'],
                           'Classifier Loss': clf_metrics['test/loss_epoch'],
                           'Accuracy': clf_metrics['test/acc_epoch']
                       }),
                       in_place=True)

       
    print(results)
    results.write_parquet('results.parquet')
    
    return None

    

if __name__ == "__main__":
    main()
