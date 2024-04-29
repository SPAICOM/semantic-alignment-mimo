"""
"""

import torch
import polars as pl
from pathlib import Path
from timm import create_model
from pytorch_lightning import Trainer
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanSquaredError
from torch.utils.data import TensorDataset, DataLoader

from src.models import RelativeEncoder, RelativeDecoder
from src.datamodules import DataModuleRelativeEncoder, DataModuleRelativeDecoder

def main() -> None:
    """The main loop. 
    """
    # Defining paths
    CURRENT: Path = Path('.')
    DATA_DIR: Path = CURRENT / "data"
    MODELS_DIR: Path = CURRENT / "models"
    
    anchors = ['10', '25', '50', '75', '100', '250', '384']
    datasets = ['cifar100']
    encoders = ['vit_base_patch16_224']
    decoders = ['vit_small_patch16_224']
    functions = ['sigmoid']
    cases = ['abs', 'abs_anch', 'rel']
    
    num_classes = 100
    batch_size = 512
    
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
                            encoder_path: Path = MODELS_DIR / f"encoders/{dataset}/{encoder}/{decoder}/{function}/{case}/anchors_{anch}.ckpt"
                            
                            # Load the encoder model
                            enc_model = RelativeEncoder.load_from_checkpoint(encoder_path)
                            
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
                            #                            Decoder Stuff
                            # =========================================================================
                            decoder_path: Path = MODELS_DIR / f"decoders/{dataset}/{decoder}/anchors_{anch}.ckpt"

                            # Load the decoder model
                            dec_model = RelativeDecoder.load_from_checkpoint(decoder_path)
                            dec_model.eval()

                            # Get and setup the decoder datamodule
                            dec_datamodule = DataModuleRelativeDecoder(dataset=dataset,
                                                                       decoder=decoder,
                                                                       num_anchors=int(anch),
                                                                       batch_size=batch_size)
                            dec_datamodule.prepare_data()
                            dec_datamodule.setup()

                            # =========================================================================
                            #                           Computing the results
                            # =========================================================================


                            # Get the relative representation in the decoder space
                            r_psi_hat = torch.cat(Trainer().predict(model=enc_model, datamodule=enc_datamodule))
                            dataloader = DataLoader(TensorDataset(r_psi_hat), batch_size=batch_size)
                           
                            # Retrieve the absolute representation in the decoder space from the r_psi_hat
                            x_psi_hat = Trainer().predict(model=dec_model, dataloaders=dataloader)

                            # Get the decoder classifier
                            clf = create_model(decoder, pretrained=True, num_classes=num_classes).get_classifier()
                            
                            # Get the prediction labels
                            clf.eval()
                            with torch.inference_mode():
                                y_hat = torch.cat([torch.argmax(clf(x_i), dim=1) for x_i in x_psi_hat])

                            x_psi_hat = torch.cat(x_psi_hat)

                            accuracy = Accuracy(task="multiclass", num_classes=num_classes)
                            mse = MeanSquaredError()
                            
                            results = results.vstack(pl.DataFrame(
                                                     {
                                                         'Dataset': dataset,
                                                         'Encoder': encoder,
                                                         'Decoder': decoder,
                                                         'Function': function,
                                                         'Case': case,
                                                         'Anchors': int(anch),
                                                         'Accuracy': accuracy(y_hat, enc_datamodule.test_data.labels).item(),
                                                         'MSE': mse(x_psi_hat, dec_datamodule.test_data.z).item()
                                                     }
                                                     ))
  
    results.write_parquet("results.parquet")

    return None

    

if __name__ == "__main__":
    main()
