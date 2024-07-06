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
        mu : float
            The mu coeficient for the admm method. Default 1e-2.
        rho : int
            The rho coeficient for the admm method. Default 1.

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
            The lambda parameter for the constraint problem.
        self.Z : torch.Tensor
            The Proximal variable for ADMM.
        self.U : torch.Tensor
            The Dual variable for ADMM.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel_matrix: torch.Tensor,
                 white_noise_cov: torch.Tensor,
                 sigma: int,
                 cost: float = None,
                 mu: float = 1e-4,
                 rho: float = 1e3):

        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix
        self.white_noise_cov = complex_tensor(white_noise_cov)
        self.sigma = sigma
        self.cost = cost
        self.mu = mu
        self.rho = rho
        self.lmb: float = None
        self.antennas_receiver, self.antennas_transmitter = self.channel_matrix.shape

        # Variables
        self.F = None
        self.G = None

        # ADMM variables
        self.Z = torch.zeros(self.antennas_transmitter, self.input_dim)
        self.U = torch.zeros(self.antennas_transmitter, self.input_dim)


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
        # self.G = self.G - 0.1 * ((self.G@self.channel_matrix@self.F@input - output)@A.H + self.G@self.white_noise_conv)
        return None
    

    def __F_step(self,
                 input: torch.Tensor,
                 output: torch.Tensor,
                 method: str = 'closed') -> None:
        """The F step that minimize the Lagrangian.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
            method : str
                The algorithm method, chose between 'admm', 'closed' and 'cvxpy'. Default 'closed'.

        Returns:
            None
        """
        def __F_admm(input: torch.Tensor,
                     output: torch.Tensor) -> None:
            """Solving the F step using admm.
            
            Args:
                input : torch.Tensor
                    The input tensor.
                output : torch.Tensor
                    The output tensor.

            Return:
                None
            """
            def __rho_update(res_primal: torch.Tensor,
                             res_dual: torch.Tensor,
                             tau_incr: float = 2,
                             tau_decr: float = 2,
                             gamma: float = 10.) -> None:
                """A function to handle the dynamic update of rho.
                
                Args:
                    res_primal : torch.Tensor
                        The primal residual.
                    res_dual : torch.Tensor
                        The dual residual.
                    tau_incr : torch.Tensor
                        The tau increase parameter. Default 2.
                    tau_decr : torch.Tensor
                        The tau decrease parameter. Default 2.
                    gamma : float
                        The gamma parameter for the check. Default 10.
                        
                Return:
                    None
                """
                res_primal = torch.linalg.matrix_norm(res_primal)
                res_dual = torch.linalg.matrix_norm(res_dual)

                if res_primal > gamma*res_dual:
                    self.rho *= tau_incr
                    self.U /= tau_incr
                elif res_dual > gamma*res_primal:
                    self.rho /= tau_decr
                    self.U *= tau_decr

                return None

            old_Z = self.Z
            
           
            # The F step
            A = self.G @ self.channel_matrix
            self.F = self.F - self.mu * ( A.H @ ( A @ self.F @ input  - output ) @ input.H + self.rho * (self.F - self.Z + self.U) )
            # self.F = self.F - self.mu * ( A.H @ ( A @ self.F @ input  - output ) @ input.H + self.rho * (self.F@input - self.Z + self.U)@input.T )

            # The Proximal randnstep
            B = self.F + self.U
            # B = self.F@input + self.U
            get_Z = lambda lmb: B/(1+lmb)
            Z = get_Z(0)
            if torch.trace(Z.H@Z).real.item() <= self.cost:
            # if torch.linalg.matrix_norm(Z).real.item() <= self.cost:
                self.Z = Z
                self.lmb = 0
            else:
                self.lmb = torch.sqrt( torch.trace(B.H@B).real / self.cost ).item()-1
                # self.lmb = torch.sqrt( torch.linalg.matrix_norm(B).real**2 / self.cost ).item()-1
                self.Z = get_Z(self.lmb)

            # print(torch.linalg.norm(self.Z).real**2)

            # The dual step
            self.U = self.U + self.F - self.Z
            # self.U = self.U + self.F@input - self.Z

            # Rho update
            res_primal = self.F - self.Z
            # res_primal = self.F@input - self.Z
            res_dual = - self.rho * (self.Z - old_Z)
            # print(res_primal, res_dual)
            # __rho_update(res_primal, res_dual)
            return None


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
            # return torch.trace((F@input).H @ (F@input)).real.item() - self.cost
            return torch.linalg.matrix_norm(F@input)**2 - self.cost
        
        input = complex_tensor(input)
        output = complex_tensor(output)
        
        assert method in ["admm", "closed", "cvxpy"], 'The passed method is not in the possible values, please chose from "admm", "closed" and "cvxpy"'
        
        # If we're in the constraint problem
        if self.cost:
            if method == 'cvxpy':
                __F_cvxpy(input, output)
            elif method == 'admm':
                __F_admm(input, output)
            elif method == 'closed':
                self.F = __F_constrained(1, input, output)
                return None
                A = self.G @ self.channel_matrix        
                self.F = torch.linalg.pinv(A) @ output @ torch.linalg.pinv(input)

                # We check if the solution with lambda = 0 is the optimum.
                self.lmb = 0

                # Check the KKT condition
                # if torch.trace((self.F@input).H @ (self.F@input)).real.item() - self.cost > 0:
                if torch.linalg.matrix_norm(self.F@input)**2 - self.cost > 0:
                    # It is not the optimum so we need to find lambda optimum
                    sol = root_scalar(__constraint, args=(input, output), x0=0, method='secant')
                    self.lmb = sol.root

                    # Get the optimal F
                    self.F = __F_constrained(self.lmb, input, output)

        # Not in the contrained problem
        else:
            A = self.G @ self.channel_matrix        
            self.F = torch.linalg.pinv(A) @ output @ torch.linalg.pinv(input)

        return None
    
    
    def fit(self,
            input: torch.Tensor,
            output: torch.Tensor,
            iterations: int = None,
            method: str = 'closed') -> list[float]:
        """Fitting the F and G to the passed data.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
            iterations : int
                The number of iterations. Default None.
            method : str
                The algorithm method, chose between 'admm', 'closed' and 'cvxpy'. Default 'closed'.
        
        Returns:
            (losses, traces) : tuple[list[float], list[float]]
                The losses and the traces during training.
        """
        input.to('cpu')
        output.to('cpu')
        
        # self.Z = torch.zeros(self.antennas_transmitter, input.T.shape[-1])
        # self.U = torch.zeros(self.antennas_transmitter, input.T.shape[-1])
        
        with torch.no_grad():
            # Inizialize the F matrix
            self.F = complex_tensor(torch.randn(self.antennas_transmitter, self.input_dim))
            # self.G = complex_tensor(torch.randn(self.output_dim, self.antennas_receiver))

            # Set the input and output in the right dimensions
            input = nn.functional.normalize(input, p=2, dim=-1)
            input = input.T
            output = output.T

            loss = np.inf
            losses = []
            traces = []
            if iterations:
                for _ in tqdm(range(iterations)):
                    self.__G_step(input=input, output=output)
                    self.__F_step(input=input, output=output, method=method)
                    # if method != 'admm':
                    #     self.__G_step(input=input, output=output)
                    # print(self.G)
                    # print(self.F)
                    loss = self.eval(input.T, output.T)
                    trace = torch.trace(self.F.H@self.F).real.item()
                    # trace = torch.trace((self.F@complex_tensor(input)).H@(self.F@complex_tensor(input))).real.item()
                    # trace = torch.linalg.matrix_norm(self.F@complex_tensor(input)).real.item()**2
                    # trace = torch.linalg.matrix_norm(self.F).real.item()**2
                    losses.append(loss)
                    traces.append(trace)
                    
                    # if self.cost:
                    #     if trace <= self.cost:
                    #         break
                    # if loss < 1e-1:
                    #     break
            else:
                while loss > 1e-1:
                    if method != 'admm':
                        self.__G_step(input=input, output=output)
                    self.__F_step(input=input, output=output, method=method)
                    loss = self.eval(input.T, output.T)
                    trace = torch.trace(self.F.H@self.F).real.item()
                    # trace = torch.trace((self.F@complex_tensor(input)).H@(self.F@complex_tensor(input))).real.item()
                    # trace = torch.linalg.matrix_norm(self.F@complex_tensor(input)).real.item()**2
                    losses.append(loss)
                    traces.append(trace)
                    
                    # if self.cost:
                    #     if trace <= self.cost:
                    #         break

        return losses, traces


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
