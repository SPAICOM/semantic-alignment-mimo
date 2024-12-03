"""This module defines the functions/classes needed for the linear optimization.
"""

import math
import torch
# import cvxpy as cp
import numpy as np
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from scipy.optimize import root_scalar
    
from src.utils import complex_tensor, complex_compressed_tensor, decompress_complex_tensor, prewhiten
# from utils import complex_tensor, complex_compressed_tensor, decompress_complex_tensor, prewhiten, snr

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


class LinearOptimizerBaseline():
    """A linear version of the baseline in which we're not taking into account the advantages of semantic communication.

    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension.
        channel_matrix : torch.Tensor
            The channel matrix H in torch.Tensor format.
        
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel_matrix: torch.Tensor,
                 sigma: int):
        
        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix
        self.sigma = sigma
        
        # Get the receiver and transmitter antennas
        self.antennas_receiver, self.antennas_transmitter = self.channel_matrix.shape

        # Instantiate the alignment matrix A
        self.A = None 

        # Perform the SVD of the channel matrix, save the U, S and Vt.
        U, S, Vt = torch.linalg.svd(self.channel_matrix)
        S = torch.diag(S).to(torch.complex64)

        # Auxiliary matrix
        B = U @ S

        # Define the decoder and precoder
        # self.G = torch.linalg.inv(S) @ U.H
        self.G = B.H @ torch.linalg.pinv(B@B.H + (1/snr(torch.ones(1), self.sigma/2)) * torch.view_as_complex(torch.stack((torch.eye(B.shape[0]), torch.eye(B.shape[0])), dim=-1)))
        self.F = Vt.H

        return None


    def __packets_precoding(self,
                            input: torch.Tensor) -> list[torch.Tensor]:
        """Precoding of packets given an input.

        Args:
            input : torch.Tensor
                The input to transmit, expected as features x number of observations.

        Returns
            list[torch.Tensor]
                The list of precoded packets, each of them of dimension self.antennas_transmitter.
        """
        input_dim, n = input.shape
        assert input_dim > self.antennas_transmitter, "The input dimension must be greater than the number of transmitting antennas"

        # Get the number of packets required for the transmission and the relative needed padding
        n_packets: int = math.ceil(input_dim / self.antennas_transmitter)
        padding_size: int = n_packets * self.antennas_transmitter - input_dim

        # Add padding rows wise (i.e. to the features)
        padded_input = torch.nn.functional.pad(input, (0, 0, 0, padding_size))
        
        # Create the packets of size self.antennas_transmitter
        packets = torch.split(padded_input, self.antennas_transmitter, dim=0)
        
        # Return the precoded packets
        return [self.F @ p for p in packets]
        

    def __packets_decoding(self,
                           packets: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decoding the transmitted packets.

        Args:
            packets : list[torch.Tensor]
                The list of the received packets.

        Returns
            received : torch.Tensor
                The output seen as a single torch.Tensor of dimension self.antennas_receiver x num. of observation.
        """
        # Decode the packets
        packets = [self.G @ p for p in packets]

        # Combined packets
        received = torch.cat(packets, dim=0)[:self.input_dim//2,:]

        return received


    def __transmit_message(self,
                           input: torch.Tensor) -> torch.Tensor:
        """Function that implements the transmission of a message.

        Args:
            input : torch.Tensor
                The input to transmit.

        Returns:
            output : torch.Tensor
                The output transmitted.
        """
        # Perform the prewhitening step
        input, L, mean = prewhiten(input)

        # Complex Compress the input
        input = complex_compressed_tensor(input.T).H
        
        # Performing the precoding packets
        packets = self.__packets_precoding(input)

        # Transmit the packets and add the AWGN
        packets = [self.channel_matrix @ p + torch.view_as_complex(torch.stack((torch.normal(0, self.sigma / 2, size=p.shape), torch.normal(0, self.sigma / 2, size=p.shape)), dim=-1)) for p in packets]

        # Decode the packets
        output = self.__packets_decoding(packets)

        # Decompress the transmitted signal
        output = decompress_complex_tensor(output.H).T

        # Remove whitening
        output = L @ output + mean

        return output.T
        

    def fit(self,
            input: torch.Tensor,
            output: torch.Tensor) -> None:
        """Fitting method of the linear baseline.
        This function performs the semantic alignment between input and output.

        Args:
            input : torch.Tensor
                The input to transmit.
            output : torch.Tensor
                The output to allign to.
            
        Returns:
            None
        """
        input.to('cpu')
        output.to('cpu')

        # Normalize the input
        input = nn.functional.normalize(input, p=2, dim=-1)

        # Alignment of the input to the output
        self.A = torch.linalg.lstsq(input, output).solution.T
        
        return None
        

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

        # Normalize the input
        input = nn.functional.normalize(input, p=2, dim=-1)

        # Align the input
        input = self.A @ input.T

        # Transmit the input
        output = self.__transmit_message(input)
                
        return output

    
    def eval(self,
             input: torch.Tensor,
             output: torch.Tensor) -> float:
        """Eval an input given an expected output.

        Returns:
            float
                The mse loss.
        """
        # Check if self.A is fitted
        assert self.A is not None, "You have to fit the solver first by calling the '.fit()' method."

        # Get the predictions
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
                 rho: float = 1e2):

        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix
        self.white_noise_cov = white_noise_cov
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
        self.Z = torch.zeros(self.antennas_transmitter, (self.input_dim + 1) //2)
        self.U = torch.zeros(self.antennas_transmitter, (self.input_dim + 1) // 2)


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
        
        A = self.channel_matrix @ self.F @ input

        # try:
        self.G = output @ A.H @ torch.linalg.inv(A @ A.H + self.white_noise_cov)  
        # self.G = output @ torch.linalg.pinv(A)  
        # except:
            # self.G = output @ A.H @ torch.linalg.pinv(A @ A.H + self.white_noise_cov)  
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
            # A = self.G @ self.channel_matrix
            A = (self.G @ self.channel_matrix).H @ (self.G @ self.channel_matrix)
            C = input @ input.H
            D = self.rho*(self.Z - self.U) + (self.G@self.channel_matrix).H @ output @ input.H

            # Being in this case C = C.H
            kr = torch.kron(C, A)

            n, m = kr.shape
            vec_F = torch.linalg.inv(kr + self.rho * torch.eye(n, m)) @ (D.H.reshape(-1))
            self.F = vec_F.reshape(list(self.F.shape)[::-1]).H
            # self.F = self.F - self.mu * ( A.H @ ( A @ self.F @ input  - output ) @ input.H + self.rho * (self.F - self.Z + self.U) )

            # The Proximal randnstep
            Z = self.F + self.U
            if torch.trace(Z.H@Z).real.item() <= self.cost:
                self.Z = Z
                self.lmb = 0
            else:
                self.lmb = torch.sqrt( torch.trace(Z.H@Z).real / self.cost ).item()-1
                self.Z = Z/(1+self.lmb)

            # The dual step
            self.U = self.U + self.F - self.Z

            # Rho update
            res_primal = self.F - self.Z
            res_dual = - self.rho * (self.Z - old_Z)
            # __rho_update(res_primal, res_dual)
            return None


        # def __F_cvxpy(input: torch.Tensor,
        #               output: torch.Tensor) -> None:
        #     """Solving the F step using cvxpy.

        #     Args:
        #         input : torch.Tensor
        #             The input tensor.
        #         output : torch.Tensor
        #             The output tensor.

        #     Return:
        #         None
        #     """
        #     F = cp.Variable(tuple(self.F.shape), complex=True)
        #     if self.F is not None:
        #         F.value = self.F.numpy()

        #     GH = (self.G @ self.channel_matrix).numpy()
        #     GH = cp.Constant(GH)
        #     cost = cp.Constant(self.cost)
        #     input = cp.Constant(input.numpy())
        #     output = cp.Constant(output.numpy())

        #     transmitted = cp.matmul(F, input)
        #     received = cp.matmul(GH, transmitted)

        #     obj = received - output
        #     norm = cp.norm(obj, 'fro')

        #     objective = cp.Minimize(norm)

        #     constraints = [cp.norm(F, 'fro') <= cost]

        #     problem = cp.Problem(objective, constraints)
        #     problem.solve(solver=cp.MOSEK, verbose=True)

        #     self.F = torch.from_numpy(F.value)
        #     return None

        
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
                vec_F = torch.linalg.inv(kr + lmb * torch.eye(n, m)) @ C.H.reshape(-1)
            except:
                vec_F = torch.linalg.pinv(kr + lmb * torch.eye(n, m)) @ C.H.reshape(-1)
                
            
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
            return torch.linalg.matrix_norm(F@input)**2 - self.cost
        
        
        assert method in ["admm", "closed", "cvxpy"], 'The passed method is not in the possible values, please chose from "admm", "closed" and "cvxpy"'
        
        # If we're in the constraint problem
        if self.cost:
            # if method == 'cvxpy':
            #     __F_cvxpy(input, output)
            if method == 'admm':
                __F_admm(input, output)
            elif method == 'closed':
                self.F = __F_constrained(1, input, output)
                return None
                A = self.G @ self.channel_matrix        
                self.F = torch.linalg.pinv(A) @ output @ torch.linalg.pinv(input)

                # We check if the solution with lambda = 0 is the optimum.
                self.lmb = 0

                # Check the KKT condition
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
            method: str = 'closed') -> tuple[list[float], list[float]]:
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
        
        with torch.no_grad():
            # Inizialize the F matrix at random
            self.F = torch.view_as_complex(torch.stack((torch.randn(self.antennas_transmitter, (self.input_dim + 1) // 2), torch.randn(self.antennas_transmitter, (self.input_dim + 1) // 2)), dim=-1))

            # Save the decompressed version
            old_input = input
            old_output = output
            
            # Set the input and output in the right dimensions
            # input = input / torch.norm(input)#nn.functional.normalize(input, p=2, dim=-1)
            input = nn.functional.normalize(input, p=2, dim=-1)
            
            input = complex_compressed_tensor(input).H
            output = complex_compressed_tensor(output).H

            loss = np.inf
            losses = []
            traces = []
            if iterations:
                for _ in tqdm(range(iterations)):
                    self.__G_step(input=input, output=output)
                    self.__F_step(input=input, output=output, method=method)
                    loss = self.eval(old_input, old_output)
                    trace = torch.trace(self.F.H@self.F).real.item()
                    losses.append(loss)
                    traces.append(trace)
                    
            else:
                while loss > 1e-1:
                    if method != 'admm':
                        self.__G_step(input=input, output=output)
                    self.__F_step(input=input, output=output, method=method)
                    loss = self.eval(old_input, old_output)
                    trace = torch.trace(self.F.H@self.F).real.item()
                    losses.append(loss)
                    traces.append(trace)
                    
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

        # Normalize the input
        # input = input / torch.norm(input)#
        input = nn.functional.normalize(input, p=2, dim=-1)

        # Complex Compress the input
        input = complex_compressed_tensor(input).H

        # Transmit the input through the channel H
        z = self.channel_matrix @ self.F @ input
        wn = torch.view_as_complex(torch.stack((torch.normal(0, self.sigma / 2, size=z.shape), torch.normal(0, self.sigma / 2, size=z.shape)), dim=-1))
        output = self.G @ (z + wn)
        
        # Decompress the transmitted signal
        output = decompress_complex_tensor(output.H)

        return output
    

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
        # preds = self.transform(input)
        
        input = nn.functional.normalize(input, p=2, dim=-1)
        # input = input / torch.norm(input)#nn.functional.normalize(input, p=2, dim=-1)
        
        input = complex_compressed_tensor(input).H
        output = complex_compressed_tensor(output).H

        return torch.trace((output - self.G @ self.channel_matrix @ self.F @ input)@(output - self.G @ self.channel_matrix @ self.F @ input).H + self.G@self.white_noise_cov@self.G.H).real.item() / (output.shape[-1]*output.shape[0]*2)
        # return torch.trace((output - self.G @ self.channel_matrix @ self.F @ input)@(output - self.G @ self.channel_matrix @ self.F @ input).H).real.item() / (output.shape[-1]*output.shape[0]*2)
        # return torch.nn.functional.mse_loss(preds, output, reduction='mean').item()



# ============================================================
#
#                     MAIN DEFINITION
#
# ============================================================

def main() -> None:
    """Some sanity tests...
    """
    from utils import complex_gaussian_matrix
    
    print("Start performing sanity tests...")
    print()
    
    # Variables definition
    input_dim = 10
    output_dim = 2
    antennas_transmitter = 4
    antennas_receiver = 4
    channel_matrix = complex_gaussian_matrix(mean=0, std=1, size=(antennas_receiver, antennas_transmitter))
    sigma = 1
    
    data = torch.randn(1000, input_dim)
    
    print("Test for Linear Optimizer Baseline...", end='\t')
    baseline = LinearOptimizerBaseline(input_dim=input_dim,
                                       output_dim=output_dim,
                                       channel_matrix=channel_matrix,
                                       sigma=sigma)
    baseline.fit(data, data)
    baseline.transform(data)
    print("[Passed]")
        
    return None


if __name__ == "__main__":
    main()
