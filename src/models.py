"""In this python module there are the models needed for the projects.
"""

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

from src.utils import complex_tensor, complex_compressed_tensor, decompress_complex_tensor, sigma_given_snr, awgn
# from utils import complex_tensor, complex_compressed_tensor, decompress_complex_tensor, sigma_given_snr, awgn


# ==================================================================
#
#                         MODELS DEFINITION
#
# ==================================================================

class ComplexAct(nn.Module):
    def __init__(self,
                 act,
                 use_phase: bool = False):
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

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.hidden_dim: int = hidden_dim
        self.hidden_size: int = hidden_size
        
        # ================================================================
        #                         Input Layer
        # ================================================================
        self.input_layer = nn.Sequential(
                                         nn.Linear(self.input_dim, self.hidden_dim),
                                         nn.GELU(),
                                         )

        # ================================================================
        #                         Hidden Layers
        # ================================================================
        self.hidden_layers = nn.ModuleList([nn.Sequential(
                                                          nn.Linear(self.hidden_dim, self.hidden_dim),
                                                          nn.GELU(),
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
    

class ComplexMLP(nn.Module):
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
        snr : float
            The snr in dB of the communication channel. Set to None if unaware. Default 20.
        cost: float
            The cost for the constrainde version. Default None.
        lmb: float
            The lambda regularizer coefficient to impose sparsity. Default 0.
        mu: float
            The mu parameter for the constrained regularization. Default 1.
        lr : float
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
                 enc_hidden_dim: int,
                 dec_hidden_dim: int,
                 hidden_size: int,
                 channel_matrix: torch.Tensor,
                 snr: float = 20.,
                 cost: float = None,
                 lmb: float = 0,
                 mu: float = 1,
                 lr: float = 1e-3):
        super().__init__()

        # Log the hyperparameters.
        self.save_hyperparameters()

        assert input_dim % 2 == 0, "The input dimension must be even."
        assert output_dim % 2 == 0, "The output dimension must be even."
        assert self.hparams["lmb"] >= 0, "The lambda parameter must be greater or equal to 0."
        assert self.hparams["mu"] >= 0, "The mu parameter must be greater or equal to 0."
        
        # Example input
        self.example_input_array = torch.randn(1, self.hparams["input_dim"])
        
        # Halve the input and output dimension
        input_dim = (input_dim + 1) // 2
        output_dim = (output_dim + 1) // 2

        self.semantic_encoder = ComplexMLP(input_dim,
                                           self.hparams["antennas_transmitter"],
                                           self.hparams["enc_hidden_dim"],
                                           self.hparams["hidden_size"])
        
        self.semantic_decoder = ComplexMLP(self.hparams["antennas_receiver"],
                                           output_dim,
                                           self.hparams["dec_hidden_dim"],
                                           self.hparams["hidden_size"])

        self.type(torch.complex64)
        
        if self.hparams["lmb"] != 0:
            for name, param in self.named_parameters():
                if param.requires_grad:  # Check only weights
                    with torch.no_grad():
                        # Freezing mask
                        layer_freeze_mask = torch.ones_like(param)
                        freeze_mask_name = name.replace('.', '__') + "_freeze_mask"
                        self.register_buffer(freeze_mask_name, layer_freeze_mask)


    def get_precodings(self,
                       x: torch.Tensor) -> torch.Tensor:
        """Get the semantic precodings of the passed tensor x:

        Args:
            x : torch.Tensor
                The input tensor x.

        Returns:
            z : torch.Tensor
                The precodings of the input tensor x.
        """
        with torch.no_grad():
            x = x.real
            x = nn.functional.normalize(x, p=2, dim=-1)

            # Complex Compression
            x = complex_compressed_tensor(x, device=self.device)

            # Precode the signal
            z = self.semantic_encoder(x)
        
        return z
    
        
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
        x = x.real
        x = nn.functional.normalize(x, p=2, dim=-1)

        x = complex_compressed_tensor(x, device=self.device)
        z = self.semantic_encoder(x)
        
        # Save the latent
        self.latent = z
        
        # Make the signal pass through the channel
        z = torch.einsum('ab, cb -> ac', z, self.hparams['channel_matrix'].to(self.device))
        
        # Add white noise
        if self.hparams["snr"]:
            sigma = sigma_given_snr(snr=self.hparams["snr"], signal=self.latent.detach())
            w = awgn(sigma=sigma, size=z.real.shape, device=self.device)
            z = z + w.detach()
            
        # Decoding in reception
        return decompress_complex_tensor(self.semantic_decoder(z), device=self.device)[:, :self.hparams['output_dim']]


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

        # Log the losses
        self.log("primal_loss", loss, on_step=True, on_epoch=True)

        if self.hparams["cost"]:
            dual_loss = ((torch.norm(self.latent, p="fro", dim=-1) - self.hparams["cost"])**2).mean()
            cost_term = self.hparams['mu'] * dual_loss            
            
            # Log the losses and trace
            self.log("dual_loss", dual_loss, on_step=True, on_epoch=True)
            self.log("cost_term", cost_term, on_step=True, on_epoch=True)
            self.log("trace", (torch.norm(self.latent, p="fro", dim=-1)**2).mean(), on_step=True, on_epoch=True)

            # Add the dual loss
            loss += cost_term

        return y_hat, loss


    def on_train_epoch_end(self) -> None:
        """Function called to apply proximal gradient descent regularization.

        Args:
            None

        Returns:
            None
        """
        if self.hparams['lmb'] != 0:
            threshold = self.hparams['lr'] * self.hparams['lmb']

            for name, param in self.named_parameters():
                if param.requires_grad and "weight" in name:  # Check only weights
                    with torch.no_grad():
                        # Create mask for weights
                        mask = torch.abs(param) <= threshold
                        param[mask] = 0.0

                        # Freezing mask
                        freeze_mask_name = name.replace('.', '__') + "_freeze_mask"
                        layer_freeze_mask = torch.ones_like(param)
                        layer_freeze_mask[mask] = 0.0
                        setattr(self, freeze_mask_name, layer_freeze_mask)
                        
                        # Find and prune corresponding bias explicitly
                        bias_name = name.replace("weight", "bias")
                        for b_name, b_param in self.named_parameters():
                            if b_name == bias_name and b_param.requires_grad:
                                # Compute row-wise AND for the mask
                                row_mask = mask.all(dim=1)  # Assuming 'mask' is 2D

                                # Apply the row mask to the bias
                                b_param[row_mask] = 0.0

                                # Freezing mask
                                freeze_mask_name = b_name.replace('.', '__') + "_freeze_mask"
                                layer_freeze_mask = torch.ones_like(b_param)
                                layer_freeze_mask[row_mask] = 0.0
                                setattr(self, freeze_mask_name, layer_freeze_mask)

        return None


    def on_after_backward(self):
        """Apply the gradient freeze mask after the backward pass.

        Args:
            None
        Returns:
            None
        """
        for name, param in self.named_parameters():
            # Check if freeze mask exists for the current parameter
            freeze_mask_name = name.replace('.', '__') + "_freeze_mask"
            if hasattr(self, freeze_mask_name):
                freeze_mask = getattr(self, freeze_mask_name)
                # Apply the freeze mask to the gradients
                if param.grad is not None:
                    param.grad *= freeze_mask

        return None


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
    input_dim = 10
    output_dim = 2
    num_classes = 2
    hidden_dim = 10
    hidden_size = 4
    activ_type = "softplus"
    antennas_transmitter = 10
    antennas_receiver = 10
    channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(antennas_receiver, antennas_transmitter))
    snr = 20
    
    data = torch.randn(10, input_dim)
    
    print()
    print("Test for Classifier...", end='\t')
    mlp = Classifier(input_dim, num_classes, hidden_dim, hidden_size)
    output = mlp(data)
    print("[Passed]")
    
    print()
    print("Test for SemanticAutoEncoder...", end='\t')
    mlp = SemanticAutoEncoder(input_dim, output_dim, antennas_transmitter, antennas_receiver, hidden_dim, hidden_dim, hidden_size, channel_matrix, snr=snr)
    output = mlp(data)
    print("[Passed]")

    return None


if __name__ == "__main__":
    main()
