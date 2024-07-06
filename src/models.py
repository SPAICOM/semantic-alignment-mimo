"""In this python module there are the models needed for the projects.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

from src.utils import complex_tensor


# ==================================================================
#
#                         MODELS DEFINITION
#
# ==================================================================

class ComplexAct(nn.Module):
    def __init__(self, act, use_phase=False):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super().__init__()
        
        self.act = act
        self.use_phase = use_phase

    def forward(self, z):
        if self.use_phase:
            return self.act(torch.abs(z)) * torch.exp(1.j * torch.angle(z)) 
        else:
            return self.act(z.real) + 1.j * self.act(z.imag)

class MLP(nn.Module):
    """An implementation of a MLP in pytorch.

    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension. Default 1.
        hidden_dim : int
            The hidden layer dimension. Default 10.
        hidden_size : int
            The number of hidden layers. Default 10.

    Attributes:
        self.<name-of-argument>:
            ex. self.input_dim is where the 'input_dim' argument is stored.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 hidden_size: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        
        # ================================================================
        #                         Input Layer
        # ================================================================
        self.input_layer = nn.Sequential(
                                         nn.Linear(self.input_dim, self.hidden_dim),
                                         ComplexAct(act=nn.GELU(), use_phase=True)
                                         )

        # ================================================================
        #                         Hidden Layers
        # ================================================================
        self.hidden_layers = nn.ModuleList([nn.Sequential(
                                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                                          ComplexAct(act=nn.GELU(), use_phase=True)
                                                          ) for _ in range(self.hidden_size)])

        # ================================================================
        #                         Output Layer
        # ================================================================
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Relative Encoder.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output of the MLP.
        """
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    

class RelativeEncoder(pl.LightningModule):
    """An implementation of a relative encoder using a MLP architecture in pytorch.

    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension. Default 1.
        hidden_dim : int
            The hidden layer dimension. Default 10.
        hidden_size : int
            The number of hidden layers. Default 10.
        activ_type : str
            The type of the last activation function. Default 'tanh'.
        lr : float)
            The learning rate. Default 1e-2. 
        momentum : float
            How much momentum to apply. Default 0.9.
        nesterov : bool
            If set to True use nesterov type of momentum. Default True.
        max_lr : float
            Maximum learning rate for the scheduler. Default 1..

    Attributes:
        self.hparams["<name-of-argument>"]:
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
                 nesterov: bool = True,
                 max_lr: float = 1.):
        super().__init__()

        # Log the hyperparameters.
        self.save_hyperparameters()

        # Example input
        self.example_input_array = torch.randn(self.hparams["input_dim"])

        # ================================================================
        #                     Activation Function    
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

        self.model = nn.Sequential(
                                       MLP(self.hparams["input_dim"],
                                           self.hparams["output_dim"],
                                           self.hparams["hidden_dim"],
                                           self.hparams["hidden_size"]),
                                       activation
                                    )
            

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Relative Encoder.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output of the MLP.
        """
        x = nn.functional.normalize(x, p=2, dim=-1)
        return self.model(x)


    def configure_optimizers(self) -> dict[str, object]:
        """Define the optimizer: Stochastic Gradient Descent.

        Returns:
            dict[str, object]
                The optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])#, momentum=self.hparams["momentum"], nesterov=self.hparams["nesterov"])
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.hparams["lr"], max_lr=self.hparams["max_lr"], step_size_up=20, mode='triangular2'),
            #     "monitor": "valid/loss_epoch"
            # }    
        }

    def loss(self,
             x: torch.Tensor,
             y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A convenient method to get the loss on a batch.

        Args:
            x : torch.Tensor
                The input tensor.
            y : torch.Tensor
                The original output tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]
                The output of the MLP and the loss.
        """
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return y_hat, loss


    def _shared_eval(self,
                     batch: list[torch.Tensor],
                     batch_idx: int,
                     prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A common step performend in the test and validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (y_hat, loss) : tuple[torch.Tensor, torch.Tensor]
                The tuple with the output of the network and the epoch loss.
        """
        x, y = batch
        y_hat, loss = self.loss(x, y)

        self.log(f'{prefix}/loss_epoch', loss, on_step=False, on_epoch=True)

        return y_hat, loss


    def training_step(self,
                      batch: list[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """The training step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            loss : torch.Tensor
                The epoch loss.
        """
        x, y = batch
        _, loss = self.loss(x, y)

        self.log('train/loss', loss, on_epoch=True)

        return loss


    def test_step(self,
                  batch: list[torch.Tensor],
                  batch_idx: int) -> None:
        """The test step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            None
        """
        _ = self._shared_eval(batch, batch_idx, "test")
        return None


    def validation_step(self,
                        batch: list[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        """The validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            y_hat : torch.Tensor
                The output of the network.
        """
        y_hat, _ = self._shared_eval(batch, batch_idx, "valid")
        return y_hat


    def predict_step(self,
                     batch: list[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx=0) -> torch.Tensor:
        """The predict step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            dataloader_idx : int
                The dataloader idx.
        
        Returns:
            torch.Tensor
                The output of the network.
        """
        x, y = batch
        return self(x)


class Classifier(pl.LightningModule):
    """An implementation of a classifier using a MLP architecture in pytorch.

    Args:
        input_dim : int
            The input dimension.
        num_classes : int
            The number of classes. Default 20.
        hidden_dim : int
            The hidden layer dimension. Default 10.
        lr : float
            The learning rate. Default 1e-2. 
        momentum : float
            How much momentum to apply. Default 0.9.
        nesterov : bool
            If set to True use nesterov type of momentum. Default True.
        max_lr : float
            Maximum learning rate for the scheduler. Default 1..

    Attributes:
        self.hparams["<name-of-argument>"]:
            ex. self.hparams["input_dim"] is where the 'input_dim' argument is stored.
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int = 20,
                 hidden_dim: int = 10,
                 lr: float = 1e-2,
                 momentum: float = 0.9,
                 nesterov: bool = True,
                 max_lr: float = 1.):
        super().__init__()

        # Log the hyperparameters.
        self.save_hyperparameters()

        self.accuracy = MulticlassAccuracy(num_classes=self.hparams["num_classes"])

        # Example input
        self.example_input_array = torch.randn(self.hparams["input_dim"])

        self.model = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.hparams['input_dim']),
            nn.Linear(self.hparams["input_dim"], self.hparams["hidden_dim"]),
            nn.Tanh(),
            nn.LayerNorm(normalized_shape=self.hparams['hidden_dim']),
            nn.Linear(self.hparams["hidden_dim"], self.hparams["num_classes"])
        )


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Classifier.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output of the MLP.
        """
        x = nn.functional.normalize(x, p=2, dim=-1)
        return self.model(x)


    def configure_optimizers(self) -> dict[str, object]:
        """Define the optimizer: Stochastic Gradient Descent.

        Returns:
            dict[str, object]
                The optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])#, momentum=self.hparams["momentum"], nesterov=self.hparams["nesterov"])
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.hparams["lr"], max_lr=self.hparams["max_lr"], step_size_up=20, mode='triangular2'),
            #     "monitor": "valid/loss_epoch"
            # }    
        }

    
    def loss(self,
             x: torch.Tensor,
             y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A convenient method to get the loss on a batch.

        Args:
            x : torch.Tensor
                The input tensor.
            y : torch.Tensor
                The original output tensor.

        Returns:
            (logits, loss) : tuple[torch.Tensor, torch.Tensor]
                The output of the MLP and the loss.
        """
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        return logits, loss


    def _shared_eval(self,
                     batch: list[torch.Tensor],
                     batch_idx: int,
                     prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A common step performend in the test and validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (logits, loss) : tuple[torch.Tensor, torch.Tensor]
                The tuple with the output of the network and the epoch loss.
        """
        x, y = batch
        logits, loss = self.loss(x, y)

        # Getting the predictions
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log(f'{prefix}/loss_epoch', loss, on_step=False, on_epoch=True)
        self.log(f'{prefix}/acc_epoch', acc, on_step=False, on_epoch=True)

        return preds, loss


    def training_step(self,
                      batch: list[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """The training step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            loss : torch.Tensor
                The epoch loss.
        """
        x, y = batch
        logits, loss = self.loss(x, y)

        # Getting the predictions
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log('train/loss', loss, on_epoch=True)
        self.log('train/acc', acc, on_epoch=True)

        return loss


    def test_step(self,
                  batch: list[torch.Tensor],
                  batch_idx: int) -> None:
        """The test step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            None
        """
        _ = self._shared_eval(batch, batch_idx, "test")
        return None


    def validation_step(self,
                        batch: list[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        """The validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            preds : torch.Tensor
                The output of the network.
        """
        preds, _ = self._shared_eval(batch, batch_idx, "valid")
        return preds


    def predict_step(self,
                     batch: list[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx=0) -> torch.Tensor:
        """The predict step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            dataloader_idx : int
                The dataloader idx.
        
        Returns:
            preds : torch.Tensor
                The output of the network.
        """
        x = batch[0]

        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        
        return preds


class SemanticAutoEncoder(pl.LightningModule):
    """An implementation of a relative encoder using a MLP architecture in pytorch.

    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension.
        antennas_transmitter : int
            The number of antennas the transmitter has.
        antennas_receiver : int
            The number of antennas the receiver has.
        hidden_dim : int
            The hidden layer dimension.
        hidden_size : int
            The number of hidden layers.
        channel_matrix : torch.Tensor
            A complex matrix simulating a communication channel.
        sigma : int
            The sigma square for the white noise. Default 0.
        lr : float)
            The learning rate. Default 1e-3. 

    Attributes:
        self.hparams["<name-of-argument>"]:
            ex. self.hparams["input_dim"] is where the 'input_dim' argument is stored.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 antennas_transmitter: int,
                 antennas_receiver: int,
                 hidden_dim: int,
                 hidden_size: int,
                 channel_matrix: torch.Tensor,
                 sigma: int = 0,
                 cost: float = None,
                 mu: float = 1e-4,
                 lr: float = 1e-3):
        super().__init__()

        # Log the hyperparameters.
        self.save_hyperparameters()

        # Example input
        self.example_input_array = torch.randn(1, self.hparams["input_dim"])

        self.semantic_encoder = MLP(self.hparams["input_dim"],
                                    self.hparams["antennas_transmitter"],
                                    self.hparams["hidden_dim"],
                                    self.hparams["hidden_size"])
        
        self.semantic_decoder = MLP(self.hparams["antennas_receiver"],
                                    self.hparams["output_dim"],
                                    self.hparams["hidden_dim"],
                                    self.hparams["hidden_size"])

        self.type(torch.complex64)

        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Relative Encoder.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output of the MLP.
        """
        x = nn.functional.normalize(x, p=2, dim=-1)

        x = complex_tensor(x)
        z = self.semantic_encoder(x)
        
        # Encoding in transmission
        # z = complex_tensor(self.semantic_encoder(x))

        # Save the latent
        self.latent = z
        
        # Make the signal pass through the channel
        z = torch.einsum('ab, cb -> ac', z, self.hparams['channel_matrix'].to(self.device))
        
        # Add white noise
        # z = z.real + torch.normal(mean=0, std=self.hparams['sigma'], size=z.real.shape).to(self.device)
        z = z + complex_tensor(torch.normal(mean=0, std=self.hparams['sigma'], size=z.real.shape).to(self.device))
        
        # Decoding in reception
        return self.semantic_decoder(z).real


    def configure_optimizers(self) -> dict[str, object]:
        """Define the optimizer: Stochastic Gradient Descent.

        Returns:
            dict[str, object]
                The optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])#, momentum=self.hparams["momentum"], nesterov=self.hparams["nesterov"])
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.hparams["lr"], max_lr=self.hparams["max_lr"], step_size_up=20, mode='triangular2'),
            #     "monitor": "valid/loss_epoch"
            # }    
        }

    def loss(self,
             x: torch.Tensor,
             y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A convenient method to get the loss on a batch.

        Args:
            x : torch.Tensor
                The input tensor.
            y : torch.Tensor
                The original output tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]
                The output of the MLP and the loss.
        """
        y_hat = self(x)
        
        loss = nn.functional.mse_loss(y_hat, y)

        if self.hparams["cost"]:
            regularizer = torch.abs(torch.norm(self.latent, p=2) - self.hparams["cost"])
            loss += self.hparams['mu'] * regularizer
            
        return y_hat, loss


    def _shared_eval(self,
                     batch: list[torch.Tensor],
                     batch_idx: int,
                     prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A common step performend in the test and validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (y_hat, loss) : tuple[torch.Tensor, torch.Tensor]
                The tuple with the output of the network and the epoch loss.
        """
        x, y = batch
        y_hat, loss = self.loss(x, y)

        self.log(f'{prefix}/loss_epoch', loss, on_step=False, on_epoch=True)

        return y_hat, loss


    def training_step(self,
                      batch: list[torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """The training step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            loss : torch.Tensor
                The epoch loss.
        """
        x, y = batch
        _, loss = self.loss(x, y)

        self.log('train/loss', loss, on_epoch=True)

        return loss


    def test_step(self,
                  batch: list[torch.Tensor],
                  batch_idx: int) -> None:
        """The test step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            None
        """
        _ = self._shared_eval(batch, batch_idx, "test")
        return None


    def validation_step(self,
                        batch: list[torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        """The validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            y_hat : torch.Tensor
                The output of the network.
        """
        y_hat, _ = self._shared_eval(batch, batch_idx, "valid")
        return y_hat


    def predict_step(self,
                     batch: list[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx=0) -> torch.Tensor:
        """The predict step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            dataloader_idx : int
                The dataloader idx.
        
        Returns:
            torch.Tensor
                The output of the network.
        """
        x, y = batch
        return self(x)


def main() -> None:
    """The main script loop in which we perform some sanity tests.
    """
    from utils import complex_gaussian_matrix
    
    print("Start performing sanity tests...")
    print()
    
    # Variables definition
    input_dim = 5
    output_dim = 2
    num_classes = 2
    hidden_dim = 10
    hidden_size = 4
    activ_type = "softplus"
    antennas_transmitter = 10
    antennas_receiver = 10
    channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(antennas_receiver, antennas_transmitter))
    sigma = 1
    
    data = torch.randn(1, input_dim)
    
    print("Test for RelativeEncoder...", end='\t')
    mlp = RelativeEncoder(input_dim, output_dim, hidden_dim, hidden_size, activ_type)
    output = mlp(data)
    print("[Passed]")

    print()
    print("Test for Classifier...", end='\t')
    mlp = Classifier(input_dim, num_classes, hidden_dim, hidden_size)
    output = mlp(data)
    print("[Passed]")
    
    print()
    print("Test for SemanticAutoEncoder...", end='\t')
    mlp = SemanticAutoEncoder(input_dim, output_dim, antennas_transmitter, antennas_receiver, hidden_dim, hidden_size, channel_matrix, sigma=sigma)
    output = mlp(data)
    print("[Passed]")

    return None


if __name__ == "__main__":
    main()
