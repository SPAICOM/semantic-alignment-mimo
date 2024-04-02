"""In this python module there is a simple implementation in pytorch of a Multi Layer Perceptron.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# ==================================================================
#
#                         MODELS DEFINITION
#
# ==================================================================

class MultiLayerPerceptron(pl.LightningModule):
    """A simple implementation of MLP in pytorch.

    Args:
        - input_dim (int): The input dimension.
        - output_dim (int): The output dimension.
        - hidden_dim (int): The hidden layer dimension.
        - hidden_size (int): The number of hidden layers.
        - lr (float): The learning rate. Default 1e-4. 

    Attributes:
        - self.hparams["<name-of-argument>"]:
            ex. self.hparams["input_dim"] is where the 'input_dim' argument is stored.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 hidden_size: int,
                 activ_type: str = "tanh",
                 lr: float = 1e-4):
        super().__init__()

        # Log the hyperparameters.
        self.save_hyperparameters()

        # Example input
        self.example_input_array = torch.randn(self.hparams["input_dim"])
        
        # Input Layer
        self.input_layer = nn.Sequential(
                                         nn.Linear(self.hparams["input_dim"], self.hparams["hidden_dim"]),
                                         nn.GELU()
                                         )

        # Hidden Layers
        self.hidden_layers = nn.ModuleList([nn.Sequential(
                                                          nn.Linear(self.hparams["hidden_dim"], self.hparams["hidden_dim"]),
                                                          nn.GELU()
                                                          ) for _ in range(self.hparams["hidden_size"])])

        # Output Layer
        # If the similarity takes values [-1, 1]
        if self.hparams["activ_type"] == "tanh":
            activation = nn.Tanh()
            
        # If the similarity takes values [0, 1]    
        elif self.hparams["activ_type"] == "sigmoid":
            activation = nn.Sigmoid()
                        
        # If the similarity is indeed a distance and takes values [0, +infinity]    
        elif self.hparams["activ_type"] == "softplus":
            activation = nn.Softplus()

        # Else return an Error
        else:
            raise Exception(f'Invalid "activ_type" passed: {self.hparams["activ_type"]} is not in the available types.')

        self.output_layer = nn.Sequential(
                                          nn.Linear(self.hparams["hidden_dim"], self.hparams["output_dim"]),
                                          activation
                                          )
            

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Multi Layer Perceptron.

        Args:
            - x (torch.Tensor): The input tensor.

        returns:
            - torch.Tensor : The output of the MLP.
        """
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


    def configure_optimizers(self):
        """Define the optimizer: Stochastic Gradient Descent.
        """
        return torch.optim.SGD(self.parameters(), lr=self.hparams["lr"])


    def loss(self,
             x: torch.Tensor,
             y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A convenient method to get the loss on a batch.

        Args:
            - x (torch.Tensor): The input tensor.
            - y (torch.Tensor): The original output tensor.

        Returns:
            - tuple[torch.Tensor, torch.Tensor] : The output of the MLP and the loss.
        """
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return y_hat, loss


    def _shared_eval(self,
                     batch,
                     batch_idx,
                     prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        """
        x, y = batch
        y_hat, loss = self.loss(x, y)

        self.log(f'{prefix}/loss_epoch', loss, on_step=False, on_epoch=True)

        return y_hat, loss


    def training_step(self,
                      batch,
                      batch_idx) -> torch.Tensor:
        """
        """
        x, y = batch
        _, loss = self.loss(x, y)

        self.log('train/loss', loss, on_epoch=True)

        return loss


    def test_step(self,
                  batch,
                  batch_idx) -> None:
        """ 
        """
        _ = self._shared_eval(batch, batch_idx, "test")
        return None


    def validation_step(self,
                        batch,
                        batch_idx) -> torch.Tensor:
        """
        """
        y_hat, _ = self._shared_eval(batch, batch_idx, "valid")
        return y_hat


    def predict_step(self,
                     batch,
                     batch_idx,
                     dataloader_idx=0) -> torch.Tensor:
        """
        """
        x, y = batch
        y_hat = self.model(x)
        return y_hat



def main():
    """The main script loop in which we perform some sanity tests.
    """
    print("Start performing sanity tests...")
    print()
    
    input_dim = 5
    output_dim = 1

    # MultiLayerPerceptron inputs
    hidden_dim = 10
    hidden_size = 4
    activ_type = "softplus"
    
    data = torch.randn(input_dim)
    
    print("Text for MultiLayerPerceptron...", end='\t')
    mlp = MultiLayerPerceptron(input_dim, output_dim, hidden_dim, hidden_size, activ_type)
    output = mlp(data)
    print("[Passed]")


if __name__ == "__main__":
    main()
