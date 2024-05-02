"""Script for getting the final parquets with the results.
"""

import torch
import polars as pl
from pathlib import Path
from timm import create_model
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy

from src.models import RelativeEncoder, Classifier
from src.datamodules import DataModuleRelativeEncoder, DataModuleClassifier

def main() -> None:
    """The main loop. 
    """
    # Defining paths
    CURRENT: Path = Path('.')
    DATA_DIR: Path = CURRENT / "data"
    MODELS_DIR: Path = CURRENT / "models"
    
    anchors = ['10', '25', '50', '75', '100', '250', '384']
    datasets = ['cifar100']
    encoders = ['vit_small_patch16_224']
    decoders = ['vit_base_patch16_224']
    functions = ['sigmoid']
    cases = ['abs']
    
    batch_size = 512
    trainer = Trainer(inference_mode=True, enable_progress_bar=False, logger=False)
    
    results = pl.DataFrame()
    for dataset in datasets:
        for encoder in encoders:
            for decoder in decoders:                
                for function in functions:
                    for case in cases:
                        for anch in anchors:

                            # =========================================================================
                            #                            Encoder Stuff
                            # =========================================================================
                            # Define the encoder model path
                            encoder_path: Path = MODELS_DIR / f"encoders/{dataset}/{encoder}/{decoder}/{function}/{case}/anchors_{anch}.ckpt"

                            # Load the encoder model
                            enc_model = RelativeEncoder.load_from_checkpoint(encoder_path)
                            enc_model.eval()
                            
                            # Get and setup the encoder datamodule
                            enc_datamodule = DataModuleRelativeEncoder(dataset=dataset,
                                                                       encoder=encoder,
                                                                       decoder=decoder,
                                                                       num_anchors=int(anch),
                                                                       case=case,
                                                                       batch_size=batch_size)
                            enc_datamodule.prepare_data()
                            enc_datamodule.setup()
                            
                            # =========================================================================
                            #                            Classfier Stuff
                            # =========================================================================
                            # Define the path towards the classifier
                            clf_path: Path = MODELS_DIR / f"decoders/{dataset}/{decoder}/anchors_{anch}.ckpt"

                            # Load the classifier model
                            clf = Classifier.load_from_checkpoint(clf_path)
                            clf.eval()

                            # Get and setup the classifier datamodule
                            clf_datamodule = DataModuleClassifier(dataset=dataset,
                                                                  decoder=decoder,
                                                                  num_anchors=int(anch),
                                                                  batch_size=batch_size)
                            clf_datamodule.prepare_data()
                            clf_datamodule.setup()

                            # =========================================================================
                            #                           Computing the results
                            # =========================================================================


                            # Get the relative representation in the decoder space
                            r_psi_hat = torch.cat(trainer.predict(model=enc_model, datamodule=enc_datamodule))
                            dataloader = DataLoader(TensorDataset(r_psi_hat), batch_size=batch_size)
                           
                            # Get the predictions using as input the r_psi_hat
                            preds = torch.cat(trainer.predict(model=clf, dataloaders=dataloader))

                            accuracy = MulticlassAccuracy(num_classes=clf_datamodule.num_classes)

                            results = results.vstack(pl.DataFrame(
                                                     {
                                                         'Dataset': dataset,
                                                         'Encoder': encoder,
                                                         'Decoder': decoder,
                                                         'Function': function,
                                                         'Case': case,
                                                         'Anchors': int(anch),
                                                         'Accuracy': accuracy(preds, enc_datamodule.test_data.labels)
                                                     }
                                                     ))
  
    results.write_parquet("results.parquet")

    # Get the original values
    original = pl.DataFrame()
    for dataset in datasets:
        for decoder in decoders:
            for anch in anchors:
                # Define the path towards the classifier
                clf_path: Path = MODELS_DIR / f"decoders/{dataset}/{decoder}/anchors_{anch}.ckpt"

                # Load the classifier model
                clf = Classifier.load_from_checkpoint(clf_path)
                clf.eval()

                # Get and setup the classifier datamodule
                clf_datamodule = DataModuleClassifier(dataset=dataset,
                                                      decoder=decoder,
                                                      num_anchors=int(anch),
                                                      batch_size=batch_size)
                clf_datamodule.prepare_data()
                clf_datamodule.setup()
                
                # Get the accuracy
                acc = trainer.test(model=clf, datamodule=clf_datamodule)[0]['test/acc_epoch']

                original = original.vstack(pl.DataFrame(
                                           {
                                               'Dataset': dataset,
                                               'Decoder': decoder,
                                               'Anchors': int(anch),
                                               'Accuracy': acc
                                           }
                                           ))

    original.write_parquet("original.parquet")

    print(results)
    print(original)
             
    return None

    

if __name__ == "__main__":
    main()
