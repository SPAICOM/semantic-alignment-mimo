"""This module defines the functions/classes needed for the linear optimization.
"""

import torch
import numpy as np
import cvxpy as cp
from pathlib import Path
from tqdm.auto import tqdm


# ============================================================
#
#                     CLASSES DEFINITION
#
# ============================================================
class LinearOptimizer():
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
                 channel_matrix: torch.Tensor):

        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix
        self.antennas_transmitter, self.antennas_receiver = self.channel_matrix.shape

        # Variables
        self.F = cp.Variable((self.antennas_transmitter, self.input_dim))
        self.G = cp.Variable((self.output_dim, self.antennas_receiver))


    def fit(self,
            input: torch.Tensor,
            output: torch.Tensor,
            iterations: int = 10) -> None:
        """Fitting the F and G to the passed data.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
            iterations : int
                The number of iterations. Default 10.
        
        Returns:
            None
        """
        F_step = torch.eye(self.antennas_transmitter, self.input_dim).numpy()

        for _ in tqdm(range(iterations)):
            # ====================================================================================
            #                                 First Iteration
            # ====================================================================================
            # Define the objective function
            first_obj = cp.Minimize(cp.norm(output.numpy().T - self.G @ self.channel_matrix.numpy() @ F_step @ input.T.numpy(), p=2))

            # Set the problem
            problem = cp.Problem(first_obj)

            # Solve the problem
            problem.solve(solver=cp.MOSEK)

            # Save the G step
            G_step = self.G.value
            
            # ====================================================================================
            #                                 Second Iteration
            # ====================================================================================
            # Define the objective function
            second_obj = cp.Minimize(cp.norm(output.numpy().T - G_step @ self.channel_matrix.numpy() @ self.F @ input.T.numpy(), p=2))

            # Set the problem
            problem = cp.Problem(second_obj)

            # Solve the problem
            problem.solve(solver=cp.MOSEK)

            # Save the F step
            F_step = self.F.value

        # Transform the solution to tensors
        self.F = torch.from_numpy(self.F.value)
        self.G = torch.from_numpy(self.G.value)

        return None


    def transform(self,
                  input: torch.Tensor) -> torch.Tensor:
        """Transform the passed input.

        Args:
            input : torch.Tensor

        Returns:
            torch.Tensor
                The transformed version of the input.
        """
        return torch.from_numpy(self.G.numpy() @ self.channel_matrix.numpy() @ self.F.numpy() @ input.T.numpy()).T.real
    

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


    def save(self,
             path: Path) -> None:
        """A function to save the F and G matrix.

        Args:
            path : Path
                The path where to save the matrixes

        Returns:
            None
        """
        torch.save(self.F, str(path / "F_matrix.pt"))
        torch.save(self.G, str(path / "G_matrix.pt"))
        return None


    def load(self,
             path: Path) -> None:
        """A function to load the saved version of F and G.

        Args:
            path : Path
                The path to the saved versions.

        Returns:
            None
        """
        self.F = torch.load(str(path / "F_matrix.pt"))
        self.G = torch.load(str(path / "G_matrix.pt"))
        return None
        
        

# ============================================================
#
#                     MAIN DEFINITION
#
# ============================================================

def main() -> None:
    """Some sanity tests...
    """
    from utils import complex_gaussian_matrix

    n = 200
    input_dim = 10
    output_dim= 15
    transmitter = 100
    receiver = 100

    inputs = torch.randn(n, input_dim)
    outputs = torch.randn(n, output_dim)

    channel_matrix = complex_gaussian_matrix(0, 1, size=(transmitter, receiver))

    opt = LinearOptimizerSAE(input_dim=input_dim, output_dim=output_dim, channel_matrix=channel_matrix)

    opt.fit(inputs, outputs)
    print(opt.eval(inputs, outputs))
        
    return None


if __name__ == "__main__":
    main()
