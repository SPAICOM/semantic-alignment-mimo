"""In this python module there are the models needed for the projects.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


# ==================================================================
#
#                         MODELS DEFINITION
#
# ==================================================================

class MultiLayerPerceptron(pl.LightningModule):
    """A simple implementation of MLP in pytorch.

    Args:
        - input_dim (int): The input dimension.
        - output_dim (int): The output dimension. Default 1.
        - hidden_dim (int): The hidden layer dimension. Default 10.
        - hidden_size (int): The number of hidden layers. Default 10.
        - activ_type (str): The type of the last activation function. Default 'tanh'.
        - lr (float): The learning rate. Default 1e-2. 

    Attributes:
        - self.hparams["<name-of-argument>"]:
            ex. self.hparams["input_dim"] is where the 'input_dim' argument is stored.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: int = 10,
                 hidden_size: int = 10,
                 activ_type: str = "tanh",
                 lr: float = 1e-2,
                 momentum: float = 0.9,
                 nesterov: bool = True):
        super().__init__()

        # Log the hyperparameters.
        self.save_hyperparameters()

        # Example input
        self.example_input_array = torch.randn(self.hparams["input_dim"])

        # ================================================================
        #                         Input Layer
        # ================================================================
        self.input_layer = nn.Sequential(
                                         nn.Linear(self.hparams["input_dim"], self.hparams["hidden_dim"]),
                                         nn.GELU()
                                         )

        # ================================================================
        #                         Hidden Layers
        # ================================================================
        self.hidden_layers = nn.ModuleList([nn.Sequential(
                                                          nn.Linear(self.hparams["hidden_dim"], self.hparams["hidden_dim"]),
                                                          nn.GELU()
                                                          ) for _ in range(self.hparams["hidden_size"])])

        # ================================================================
        #                         Output Layer
        # ================================================================
        # If the similarity takes values [-1, 1]
        if self.hparams["activ_type"] == "tanh":
            activation = nn.Tanh()
            
        # If the similarity takes values [0, 1]    
        elif self.hparams["activ_type"] == "sigmoid":
            activation = nn.Sigmoid()
                        
        # If the similarity is indeed a distance and takes values [0, +infinity]    
        elif self.hparams["activ_type"] == "softplus":
            activation = nn.Softplus()
        elif self.hparams["activ_type"] == "relu":
            activation = nn.ReLU()

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
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor : The output of the MLP.
        """
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Define the optimizer: Stochastic Gradient Descent.
        """
        return torch.optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"], nesterov=self.hparams["nesterov"])


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
                     batch: torch.Tensor,
                     batch_idx: int,
                     prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A common step performend in the test and validation step.

        Args:
            - batch (torch.Tensor): The current batch.
            - batch_idx (int): The batch index.
            - prefix (str): The step type for logging purposes.

        Returns:
            - (y_hat, loss) (tuple[torch.Tensor, torch.Tensor]): The tuple with the output of the network and the epoch loss.
        """
        x, y = batch
        y_hat, loss = self.loss(x, y)

        self.log(f'{prefix}/loss_epoch', loss, on_step=False, on_epoch=True)

        return y_hat, loss


    def training_step(self,
                      batch: torch.Tensor,
                      batch_idx: int) -> torch.Tensor:
        """The training step.

        Args:
            - batch (torch.Tensor): The current batch.
            - batch_idx (int): The batch index.

        Returns:
            - loss (torch.Tensor): The epoch loss.
        """
        x, y = batch
        _, loss = self.loss(x, y)

        self.log('train/loss', loss, on_epoch=True)

        return loss


    def test_step(self,
                  batch: torch.Tensor,
                  batch_idx: int) -> None:
        """The test step.

        Args:
            - batch (torch.Tensor): The current batch.
            - batch_idx (int): The batch index.

        Returns:
            - None
        """
        _ = self._shared_eval(batch, batch_idx, "test")
        return None


    def validation_step(self,
                        batch: torch.Tensor,
                        batch_idx: int) -> torch.Tensor:
        """The validation step.

        Args:
            - batch (torch.Tensor): The current batch.
            - batch_idx (int): The batch index.

        Returns:
            - y_hat (torch.Tensor): The output of the network.
        """
        y_hat, _ = self._shared_eval(batch, batch_idx, "valid")
        return y_hat


    def predict_step(self,
                     batch: torch.Tensor,
                     batch_idx: int,
                     dataloader_idx=0) -> torch.Tensor:
        """The predict step.

        Args:
            - batch (torch.Tensor): The current batch.
            - batch_idx (int): The batch index.

        Returns:
            - y_hat (torch.Tensor): The output of the network.
        """
        x, y = batch
        y_hat = self.model(x)
        return y_hat



class RelativeDecoder(pl.LightningModule):
    """An implementation of a relative decoder using a MLP architecture in pytorch.

    Args:
        - input_dim (int): The input dimension.
        - output_dim (int): The output dimension. Default 1.
        - hidden_dim (int): The hidden layer dimension. Default 10.
        - hidden_size (int): The number of hidden layers. Default 10.
        - lr (float): The learning rate. Default 1e-2. 

    Attributes:
        - self.hparams["<name-of-argument>"]:
            ex. self.hparams["input_dim"] is where the 'input_dim' argument is stored.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: int = 10,
                 hidden_size: int = 10,
                 lr: float = 1e-2,
                 momentum: float = 0.9,
                 nesterov: bool = True):
        super().__init__()

        # Log the hyperparameters.
        self.save_hyperparameters()

        # Example input
        self.example_input_array = torch.randn(self.hparams["input_dim"])

        # ================================================================
        #                         Input Layer
        # ================================================================
        self.input_layer = nn.Sequential(
                                         nn.Linear(self.hparams["input_dim"], self.hparams["hidden_dim"]),
                                         nn.GELU()
                                         )

        # ================================================================
        #                         Hidden Layers
        # ================================================================
        self.hidden_layers = nn.ModuleList([nn.Sequential(
                                                          nn.Linear(self.hparams["hidden_dim"], self.hparams["hidden_dim"]),
                                                          nn.GELU()
                                                          ) for _ in range(self.hparams["hidden_size"])])

        # ================================================================
        #                         Output Layer
        # ================================================================
        self.output_layer = nn.Linear(self.hparams["hidden_dim"], self.hparams["output_dim"])


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


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Define the optimizer: Stochastic Gradient Descent.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"], nesterov=self.hparams["nesterov"])
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=0.1, max_lr=0.9, mode='exp_range') 

        return [optimizer], [scheduler]


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
                     batch: torch.Tensor,
                     batch_idx: int,
                     prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A common step performend in the test and validation step.

        Args:
            - batch (torch.Tensor): The current batch.
            - batch_idx (int): The batch index.
            - prefix (str): The step type for logging purposes.

        Returns:
            - (y_hat, loss) (tuple[torch.Tensor, torch.Tensor]): The tuple with the output of the network and the epoch loss.
        """
        x, y = batch
        y_hat, loss = self.loss(x, y)

        self.log(f'{prefix}/loss_epoch', loss, on_step=False, on_epoch=True)

        return y_hat, loss


    def training_step(self,
                      batch: torch.Tensor,
                      batch_idx: int) -> torch.Tensor:
        """The training step.

        Args:
            - batch (torch.Tensor): The current batch.
            - batch_idx (int): The batch index.

        Returns:
            - loss (torch.Tensor): The epoch loss.
        """
        x, y = batch
        _, loss = self.loss(x, y)

        self.log('train/loss', loss, on_epoch=True)

        return loss


    def test_step(self,
                  batch: torch.Tensor,
                  batch_idx: int) -> None:
        """The test step.

        Args:
            - batch (torch.Tensor): The current batch.
            - batch_idx (int): The batch index.

        Returns:
            - None
        """
        _ = self._shared_eval(batch, batch_idx, "test")
        return None


    def validation_step(self,
                        batch: torch.Tensor,
                        batch_idx: int) -> torch.Tensor:
        """The validation step.

        Args:
            - batch (torch.Tensor): The current batch.
            - batch_idx (int): The batch index.

        Returns:
            - y_hat (torch.Tensor): The output of the network.
        """
        y_hat, _ = self._shared_eval(batch, batch_idx, "valid")
        return y_hat


    def predict_step(self,
                     batch: torch.Tensor,
                     batch_idx: int,
                     dataloader_idx=0) -> torch.Tensor:
        """The predict step.

        Args:
            - batch (torch.Tensor): The current batch.
            - batch_idx (int): The batch index.

        Returns:
            - y_hat (torch.Tensor): The output of the network.
        """
        x, y = batch
        y_hat = self.model(x)
        return y_hat



def main() -> None:
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
    
    print("Test for MultiLayerPerceptron...", end='\t')
    mlp = MultiLayerPerceptron(input_dim, output_dim, hidden_dim, hidden_size, activ_type)
    output = mlp(data)
    print("[Passed]")

    print("Test for RelativeDecoder...", end='\t')
    mlp = RelativeDecoder(input_dim, output_dim, hidden_dim, hidden_size)
    output = mlp(data)
    print("[Passed]")

    return None


if __name__ == "__main__":
    main()
