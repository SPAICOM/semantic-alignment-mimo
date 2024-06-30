"""This module defines the functions/classes needed for the linear optimization.
"""

import torch
import cvxpy as cp
import numpy as np
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from scipy.optimize import root_scalar
    
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
        cost : float
            The transmition cost. Default None.

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
        self.lmb : float
            The lambda parameter for the constraint problem
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel_matrix: torch.Tensor,
                 white_noise_cov: torch.Tensor,
                 sigma: int,
                 cost: float = None):

        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix
        self.white_noise_cov = white_noise_cov
        self.sigma = sigma
        self.cost = cost
        self.lmb: float = None
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

        try:
            self.G = output @ A.H @ torch.linalg.inv(A @ A.H + self.white_noise_cov)  
        except:
            self.G = output @ A.H @ torch.linalg.pinv(A @ A.H + self.white_noise_cov)  
            
        
        return None
    

    def __F_step(self,
                 input: torch.Tensor,
                 output: torch.Tensor,
                 cvxpy: bool = False) -> None:
        """The F step that minimize the Lagrangian.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
            cvxpy : bool
                Check for using cvxpy solvers.

        Returns:
            None
        """
        def __F_cvxpy(input: torch.Tensor,
                      output: torch.Tensor) -> None:
            """Solving the F step using cvxpy.

            Args:
                input : torch.Tensor
                    The input tensor.
                output : torch.Tensor
                    The output tensor.

            Return:
                None
            """
            F = cp.Variable(tuple(self.F.shape), complex=True)
            if self.F is not None:
                F.value = self.F.numpy()

            GH = (self.G @ self.channel_matrix).numpy()
            GH = cp.Constant(GH)
            cost = cp.Constant(self.cost)
            input = cp.Constant(input.numpy())
            output = cp.Constant(output.numpy())

            transmitted = cp.matmul(F, input)
            received = cp.matmul(GH, transmitted)

            obj = received - output
            norm = cp.norm(obj, 'fro')

            objective = cp.Minimize(norm)

            constraints = [cp.norm(F, 'fro') <= cost]

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK, verbose=True)

            self.F = torch.from_numpy(F.value)
            return None

        
        def __F_constrained(lmb: float,
                            input: torch.Tensor,
                            output: torch.Tensor) -> torch.Tensor:
            """F when the problem is constrained.

            Args:
                lmb : float
                    The KKT parameter.
                input : torch.Tensor
                    The input tensor.
                output : torch.Tensor
                    The output tensor.

            Returns:
                torch.Tensor
                    The founded F.
            """
            A = (self.G @ self.channel_matrix).H @ (self.G @ self.channel_matrix)
            B = input @ input.H
            C = (self.G @ self.channel_matrix).H @ output @ input.H

            # Remembering that in this case B = B.H
            kr = torch.kron(B, A)
            n, m = kr.shape

            try:
                vec_F = torch.linalg.inv(kr + lmb.item() * torch.eye(n, m)) @ C.H.reshape(-1)
            except:
                vec_F = torch.linalg.pinv(kr + lmb.item() * torch.eye(n, m)) @ C.H.reshape(-1)
                
            
            return vec_F.reshape(list(self.F.shape)[::-1]).H
        
        
        def __constraint(lmb: float,
                         input: torch.Tensor,
                         output: torch.Tensor) -> float:
            """The constraint.

            Args:
                lmb : float
                    The KKT parameter.
                input : torch.Tensor
                    The input tensor.
                output : torch.Tensor
                    The output tensor.

            Returns:
                float
                    The contraint value.
            """
            F = __F_constrained(lmb, input, output)
            return torch.trace(F.H @ F).real.item() - self.cost
        
        input = complex_tensor(input)
        output = complex_tensor(output)

        if cvxpy:
            __F_cvxpy(input, output)
        else:
            self.lmb = None
            A = self.G @ self.channel_matrix        
            self.F = torch.linalg.pinv(A) @ output @ torch.linalg.pinv(input)

            # If we're in the constraint problem
            if self.cost:

                # We check if the solution with lambda = 0 is the optimum.
                self.lmb = 0

                # Check the KKT condition
                if torch.trace(self.F.H @ self.F).real.item() - self.cost > 0:
                    # It is not the optimum so we need to find lambda optimum
                    sol = root_scalar(__constraint, args=(input, output), x0=0, method='secant')
                    self.lmb = sol.root

                    # Get the optimal F
                    self.F = __F_constrained(self.lmb, input, output)
        
        return None
    
    
    def fit(self,
            input: torch.Tensor,
            output: torch.Tensor,
            iterations: int = None,
            cvxpy: bool = False) -> list[float]:
        """Fitting the F and G to the passed data.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
            iterations : int
                The number of iterations. Default None.
            cvxpy : bool
                Check for using cvxpy solvers.
        
        Returns:
            loss : list[float]
                The list of all the losses during fit.
        """
        input.to('cpu')
        output.to('cpu')
        
        with torch.no_grad():
            # Inizialize the F matrix
            self.F = complex_tensor(torch.randn(self.antennas_transmitter, self.input_dim))

            # Set the input and output in the right dimensions
            input = nn.functional.normalize(input, p=2, dim=-1)
            input = input.T
            output = output.T

            loss = np.inf
            losses = []
            if iterations:
                for _ in tqdm(range(iterations)):
                    self.__G_step(input=input, output=output)
                    self.__F_step(input=input, output=output, cvxpy=cvxpy)
                    loss = self.eval(input.T, output.T)
                    losses.append(loss)
                    
                    # if loss < 1e-1:
                    #     break
            else:
                while loss > 1e-1:
                    self.__G_step(input=input, output=output)
                    self.__F_step(input=input, output=output, cvxpy=cvxpy)
                    loss = self.eval(input.T, output.T)
                    losses.append(loss)

        return losses


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
        input.to('cpu')
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
        input.to('cpu')
        output.to('cpu')
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
