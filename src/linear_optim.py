"""This module defines the functions/classes needed for the linear optimization.
"""

import torch
import numpy as np
import cvxpy as cp
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm

from src.utils import complex_tensor

# ============================================================
#
#                     CLASSES DEFINITION
#
# ============================================================
class LinearOptimizerRE():
    """The linear optimizer for the Relative Decoder.
    
    Args:
        solver : str
            The type of solver strategy used. Choose between ['ortho', 'free']. Default 'ortho'.

    Attributes:
        self.<args_name>
        self.W : torch.Tensor
            The matrix that solves the linear problem.
    """
    def __init__(self,
                 solver: str = 'ortho'):
        assert solver in ['ortho', 'free'], "The passed solver is not one of the possible values. Choose between ['ortho', 'free']"
        
        self.solver = solver
        self.W = None

    
    def fit(self,
            input: torch.Tensor,
            output: torch.Tensor) -> None:
        """Fitting the W to the passed data.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
        
        Returns:
            None
        """
        input = torch.nn.functional.normalize(input, p=2, dim=-1)
        
        if self.solver == "ortho":
           
            u, _, v = torch.svd(input.T@output)
            self.W = u@v.T 
            
        elif self.solver == "free":
            
            A = 2 * input.T@input
            B = input.T@output
            self.W = torch.linalg.solve(A, B)
            
        else:
            raise Exception(f"The passed 'solver' is not in the possible values. Passed 'solver'={self.solver} which is not 'ortho' or 'free'.")

        return None
        

    def transform(self,
                  input: torch.Tensor) -> torch.Tensor:
        """Transform the passed input.

        Args:
            input : torch.Tensor

        Returns:
            torch.Tensor
                The input@self.W version output.
        """
        assert self.W is not None, "The 'W' matrix is currently None. You have to first fit the Linear model."
        input = torch.nn.functional.normalize(input, p=2, dim=-1)
        return input@self.W


    def eval(self,
             input: torch.Tensor,
             output: torch.Tensor) -> float:
        """Eval an input given an expected output.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
        
        Returns:
            float
                The mse loss.
        """
        preds = self.transform(input)
        return torch.nn.functional.mse_loss(preds, output, reduction='mean').item()

    
class LinearOptimizerSAE():
    """The linear optimizer for the Semantic Auto Encoder.
    
    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension.
        channel_matrix : torch.Tensor
            The Complex Gaussian Matrix simulating the communication channel.
        white_noise_cov: torch.Tensor
            The covariance matrix of white noise.
        sigma : int
            The sigma square for the white noise.

    Attributes:
        self.<args_name>
        self.antennas_transmitter : int
            The number of antennas transmitting the signal.
        self.antennas_receiver : int
            The number of antennas receiving the signal.
        self.F : cp.Variable | torch.Tensor
            The F matrix.
        self.G : cp.Variable | torch.Tensor
            The G matrix.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel_matrix: torch.Tensor,
                 white_noise_cov: torch.Tensor,
                 sigma: int):

        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix
        self.white_noise_cov = white_noise_cov
        self.sigma = sigma
        self.antennas_receiver, self.antennas_transmitter = self.channel_matrix.shape

        # Variables
        self.F = None
        self.G = None


    def __G_step(self,
                 input: torch.Tensor,
                 output: torch.Tensor) -> None:
        """The G step that minimize the Lagrangian.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.

        Returns:
            None
        """
        input = complex_tensor(input)
        output = complex_tensor(output)
        
        A = self.channel_matrix @ self.F @ input
        self.G = output @ A.H @ torch.linalg.inv(A @ A.H + self.white_noise_cov)  
        
        return None
    

    def __F_step(self,
                 input: torch.Tensor,
                 output: torch.Tensor) -> None:
        """The F step that minimize the Lagrangian.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.

        Returns:
            None
        """
        input = complex_tensor(input)
        output = complex_tensor(output)
        
        A = self.G @ self.channel_matrix
        self.F = torch.linalg.pinv(A) @ output @ torch.linalg.pinv(input)
        
        return None
    
    
    def fit(self,
            input: torch.Tensor,
            output: torch.Tensor,
            iterations: int = 10,
            eval: bool = False) -> list[float]:
        """Fitting the F and G to the passed data.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
            iterations : int
                The number of iterations. Default 10.
            eval: bool
                Set to True if you want to eval every single iteration. Default False.
        
        Returns:
            loss : list[float]
                The list of all the losses during fit.
        """
        # Inizialize the F matrix
        self.F = complex_tensor(torch.randn(self.antennas_transmitter, self.input_dim))

        # Set the input and output in the right dimensions
        input = nn.functional.normalize(input, p=2, dim=-1)
        input = input.T
        output = output.T

        loss = []
        for _ in tqdm(range(iterations)):
            self.__G_step(input=input, output=output)
            self.__F_step(input=input, output=output)
            if eval:
                loss.append(self.eval(input.T, output.T))

        return loss


    def transform(self,
                  input: torch.Tensor) -> torch.Tensor:
        """Transform the passed input.

        Args:
            input : torch.Tensor
                The input tensor.

        Returns:
            output : torch.Tensor
                The transformed version of the input.
        """
        input = nn.functional.normalize(input, p=2, dim=-1)
        input = complex_tensor(input.T)
        z = self.channel_matrix @ self.F @ input
        wn = complex_tensor(torch.normal(0, self.sigma, size=z.shape))
        output = self.G @ (z + wn)

        return output.real.T
    

    def eval(self,
             input: torch.Tensor,
             output: torch.Tensor) -> float:
        """Eval an input given an expected output.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
        
        Returns:
            float
                The mse loss.
        """
        preds = self.transform(input)
        return torch.nn.functional.mse_loss(preds, output, reduction='mean').item()



# ============================================================
#
#                     MAIN DEFINITION
#
# ============================================================

def main() -> None:
    """Some sanity tests...
    """
    return None


if __name__ == "__main__":
    main()
