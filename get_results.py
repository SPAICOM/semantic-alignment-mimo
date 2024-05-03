"""Script for getting the final parquet with the results.
"""

import torch
import polars as pl
from pathlib import Path
from timm import create_model
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader

from src.models import RelativeEncoder, Classifier
from src.datamodules import DataModuleRelativeEncoder, DataModuleClassifier

def main() -> None:
    """The main loop. 
    """
    # Defining paths
    CURRENT: Path = Path('.')
    DATA_DIR: Path = CURRENT / "data"
    MODELS_DIR: Path = CURRENT / "models"
    
    batch_size = 512
    trainer = Trainer(inference_mode=True, enable_progress_bar=False, logger=False)
    
    results = pl.DataFrame()
    for encoder_path in (MODELS_DIR / 'encoders').rglob('*.ckpt'):
        # Getting the settings
        _, _, dataset, encoder, decoder, function, case, ckpt = str(encoder_path).split("\\")
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
        clf_path: Path = MODELS_DIR / f"decoders/{dataset}/{decoder}/{function}/anchors_{anchors}.ckpt"

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
        
        results = results.vstack(pl.DataFrame(
                                 {
                                     'Dataset': dataset,
                                     'Encoder': encoder,
                                     'Decoder': decoder,
                                     'Function': function,
                                     'Case': case,
                                     'Anchors': anchors,
                                     'Alignment Loss': alignment_metrics['test/loss_epoch'],
                                     'Classifier Loss': clf_metrics['test/loss_epoch'],
                                     'Accuracy': clf_metrics['test/acc_epoch']
                                 }
                                 ))
  
    # ==============================================================
    #                Get the original values
    # ==============================================================
    for clf_path in (MODELS_DIR / 'decoders').rglob('*.ckpt'):
        # Getting the settings
        _, _, dataset, decoder, function, ckpt = str(clf_path).split("\\")
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
        
        results = results.vstack(pl.DataFrame(
                                   {
                                       'Dataset': dataset,
                                       'Encoder': decoder,
                                       'Decoder': decoder,
                                       'Function': function,
                                       'Case': 'Original Model',
                                       'Anchors': anchors,
                                       'Alignment Loss': None,
                                       'Classifier Loss': metrics['test/loss_epoch'],
                                       'Accuracy': metrics['test/acc_epoch']
                                   }
                                   ))

    print(results)
    results.write_parquet('results.parquet')
    
    return None

    

if __name__ == "__main__":
    main()
