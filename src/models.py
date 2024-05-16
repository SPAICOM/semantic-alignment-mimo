"""In this python module there are the models needed for the projects.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy


# ==================================================================
#
#                         MODELS DEFINITION
#
# ==================================================================

class RelativeEncoder(pl.LightningModule):
    """An implementation of a relative encoder using a MLP architecture in pytorch.

    Args:
        input_dim (int): The input dimension.
        output_dim (int): The output dimension. Default 1.
        hidden_dim (int): The hidden layer dimension. Default 10.
        hidden_size (int): The number of hidden layers. Default 10.
        activ_type (str): The type of the last activation function. Default 'tanh'.
        lr (float): The learning rate. Default 1e-2. 
        momentum (float): How much momentum to apply. Default 0.9.
        nesterov (bool): If set to True use nesterov type of momentum. Default True.
        max_lr (float): Maximum learning rate for the scheduler. Default 1..

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
        """The forward pass of the Relative Encoder.

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


    def configure_optimizers(self) -> dict[str, object]:
        """Define the optimizer: Stochastic Gradient Descent.

        Returns:
            dict[str, object] : The optimizer and scheduler.
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
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The original output tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor] : The output of the MLP and the loss.
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
            batch (list[torch.Tensor]): The current batch.
            batch_idx (int): The batch index.
            prefix (str): The step type for logging purposes.

        Returns:
            (y_hat, loss) (tuple[torch.Tensor, torch.Tensor]): The tuple with the output of the network and the epoch loss.
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
            batch (list[torch.Tensor]): The current batch.
            batch_idx (int): The batch index.

        Returns:
            loss (torch.Tensor): The epoch loss.
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
            batch (list[torch.Tensor]): The current batch.
            batch_idx (int): The batch index.

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
            batch (list[torch.Tensor]): The current batch.
            batch_idx (int): The batch index.

        Returns:
            y_hat (torch.Tensor): The output of the network.
        """
        y_hat, _ = self._shared_eval(batch, batch_idx, "valid")
        return y_hat


    def predict_step(self,
                     batch: list[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx=0) -> torch.Tensor:
        """The predict step.

        Args:
            batch (list[torch.Tensor]): The current batch.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader idx.
        
        Returns:
            (torch.Tensor): The output of the network.
        """
        x, y = batch
        return self(x)


class Classifier(pl.LightningModule):
    """An implementation of a classifier using a MLP architecture in pytorch.

    Args:
        input_dim (int): The input dimension.
        num_classes (int): The number of classes. Default 20.
        hidden_dim (int): The hidden layer dimension. Default 10.
        lr (float): The learning rate. Default 1e-2. 
        momentum (float): How much momentum to apply. Default 0.9.
        nesterov (bool): If set to True use nesterov type of momentum. Default True.
        max_lr (float): Maximum learning rate for the scheduler. Default 1..

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
            # self.Lambda(lambda x: x.permute(1, 0)),
            # nn.InstanceNorm1d(self.hparams["hidden_dim"]),
            # self.Lambda(lambda x: x.permute(1, 0)),
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
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The original output tensor.

        Returns:
            (logits, loss) tuple[torch.Tensor, torch.Tensor] : The output of the MLP and the loss.
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
            batch (list[torch.Tensor]): The current batch.
            batch_idx (int): The batch index.
            prefix (str): The step type for logging purposes.

        Returns:
            (logits, loss) (tuple[torch.Tensor, torch.Tensor]): The tuple with the output of the network and the epoch loss.
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
            batch (list[torch.Tensor]): The current batch.
            batch_idx (int): The batch index.

        Returns:
            loss (torch.Tensor): The epoch loss.
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
            batch (list[torch.Tensor]): The current batch.
            batch_idx (int): The batch index.

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
            batch (list[torch.Tensor]): The current batch.
            batch_idx (int): The batch index.

        Returns:
            preds (torch.Tensor): The output of the network.
        """
        preds, _ = self._shared_eval(batch, batch_idx, "valid")
        return preds


    def predict_step(self,
                     batch: list[torch.Tensor],
                     batch_idx: int,
                     dataloader_idx=0) -> torch.Tensor:
        """The predict step.

        Args:
            batch (list[torch.Tensor]): The current batch.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader idx.
        
        Returns:
            preds (torch.Tensor): The output of the network.
        """
        x = batch[0]

        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        
        return preds


    class Lambda(nn.Module):
        """An inner class defined at 'https://github.com/lucmos/relreps/blob/main/experiments/sec%3Amodel-reusability-vision/vision_stitching_cifar100.ipynb'.
        """
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self,
                    x: torch.Tensor) -> torch.Tensor:
            """The forward pass.

            Args:
                x : torch.Tensor
                    The input tensor.

            Returns:
                torch.Tensor
                    The output in tensor format.
            """
            return self.func(x)



def main() -> None:
    """The main script loop in which we perform some sanity tests.
    """
    print("Start performing sanity tests...")
    print()
    
    input_dim = 5
    samples = 2
    output_dim = 2
    num_classes = 2

    # RelativeEncoder inputs
    hidden_dim = 10
    hidden_size = 4
    activ_type = "softplus"
    
    data = torch.randn(samples, input_dim)
    
    print("Test for RelativeEncoder...", end='\t')
    mlp = RelativeEncoder(input_dim, output_dim, hidden_dim, hidden_size, activ_type)
    output = mlp(data)
    print("[Passed]")

    print("Test for Classifier...", end='\t')
    mlp = Classifier(input_dim, num_classes, hidden_dim, hidden_size)
    output = mlp(data)
    print("[Passed]")

    return None


if __name__ == "__main__":
    main()
