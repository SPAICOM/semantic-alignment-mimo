"""This module defines the functions/classes needed for the linear optimization.
"""

import math
import torch
import numpy as np
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
    
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
        self.G = B.H @ torch.linalg.inv(B@B.H + (1/snr(torch.ones(1), self.sigma/2)) * torch.view_as_complex(torch.stack((torch.eye(B.shape[0]), torch.eye(B.shape[0])), dim=-1)))
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
        with torch.no_grad():
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

        with torch.no_grad():
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

        # Transpose
        input = input.T

        # Align the input
        input = self.A @ input

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
        input = input.to("cpu")
        output = output.to("cpu")
        
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
            The transmition cost. Default 1.0.
        rho : int
            The rho coeficient for the admm method. Default 1.

    Attributes:
        self.<args_name>
        self.antennas_transmitter : int
            The number of antennas transmitting the signal.
        self.antennas_receiver : int
            The number of antennas receiving the signal.
        self.F : torch.Tensor
            The F matrix.
        self.G : torch.Tensor
            The G matrix.
        self.Z : torch.Tensor
            The Proximal variable for ADMM.
        self.U : torch.Tensor
            The Dual variable for ADMM.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel_matrix: torch.Tensor,
                 sigma: int,
                 cost: float = 1.0,
                 rho: float = 1e2):

        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix
        self.sigma = sigma
        self.cost = cost
        self.rho = rho
        self.antennas_receiver, self.antennas_transmitter = self.channel_matrix.shape

        # Variables
        self.F = None
        self.G = None

        # ADMM variables
        self.Z = torch.zeros(self.antennas_transmitter, (self.input_dim + 1) // 2)
        self.U = torch.zeros(self.antennas_transmitter, (self.input_dim + 1) // 2)

        return None
    

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
        # Get the auxiliary matrix A
        A = self.channel_matrix @ self.F @ input

        # try:
        self.G = output @ A.H @ torch.linalg.inv(A @ A.H + (self.sigma/2)* torch.view_as_complex(torch.stack((torch.eye(A.shape[0]), torch.eye(A.shape[0])), dim=-1)))
        # except:
            # self.G = output @ A.H @ torch.linalg.pinv(A @ A.H + self.white_noise_cov)  
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
        # Get the auxiliary matrixes
        O = self.G @ self.channel_matrix
        A = O.H @ O
        Bh = (input @ input.H).H
        C = self.rho * (self.Z - self.U) + O.H @ output @ input.H

        kron = torch.kron(Bh.contiguous(), A.contiguous()) 
        n, m = kron.shape
        
        vec_F = torch.linalg.inv(kron + self.rho * torch.eye(n, m)) @ C.T.reshape(-1)
        self.F = vec_F.reshape(self.F.T.shape).T
    
        return None


    def __Z_step(self) -> None:
        """The Z step for the Scaled ADMM.

        Args:
            None
                
        Returns:
            None
        """
        # Get the auxiliary matrix C
        C = self.F + self.U
        tr = torch.trace(C @ C.H).real

        if tr <= self.cost:
            self.Z = C
        else:
            lmb = torch.sqrt(tr / self.cost).item() -1
            self.Z = C / (1 + lmb)
            
        return None


    def __U_step(self) -> None:
        """The U step for the Scaled ADMM.

        Args:
            None
            
        Returns:
            None
        """
        self.U = self.U + self.F - self.Z
        return None
    
    
    def fit(self,
            input: torch.Tensor,
            output: torch.Tensor,
            iterations: int = None) -> tuple[list[float], list[float]]:
        """Fitting the F and G to the passed data.

        Args:
            input : torch.Tensor
                The input tensor.
            output : torch.Tensor
                The output tensor.
            iterations : int
                The number of iterations. Default None.
        
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
            input = nn.functional.normalize(input, p=2, dim=-1)

            # Transpose
            input = input.T
            output = output.T

            # Perform the prewhitening step
            input, _, _, output = prewhiten(input, output)

            # Complex compression
            input = complex_compressed_tensor(input.T).H
            output = complex_compressed_tensor(output.T).H

            loss = np.inf
            losses = []
            traces = []
            if iterations:
                for _ in tqdm(range(iterations)):
                    self.__G_step(input=input, output=output)
                    self.__F_step(input=input, output=output)
                    self.__Z_step()
                    self.__U_step()
                    loss = self.eval(old_input, old_output)
                    trace = torch.trace(self.F.H@self.F).real.item()
                    losses.append(loss)
                    traces.append(trace)
                    
            else:
                while loss > 1e-1:
                    self.__G_step(input=input, output=output)
                    self.__F_step(input=input, output=output)
                    self.__Z_step()
                    self.__U_step()
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

        with torch.no_grad():
            # Normalize the input
            input = nn.functional.normalize(input, p=2, dim=-1)

            # Transpose
            input = input.T

            # Perform the prewhitening step
            input, L, mean = prewhiten(input)
        
            # Complex Compress the input
            input = complex_compressed_tensor(input.T).H

            # Transmit the input through the channel H
            z = self.channel_matrix @ self.F @ input
            wn = torch.view_as_complex(torch.stack((torch.normal(0, self.sigma / 2, size=z.shape), torch.normal(0, self.sigma / 2, size=z.shape)), dim=-1))
            output = self.G @ (z + wn)
        
            # Decompress the transmitted signal
            output = decompress_complex_tensor(output.H).T
        
            # Remove whitening
            output = L @ output + mean

        return output.T
    

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

        # Check if self.F and self.G are fitted
        assert (self.F is not None)&(self.G is not None), "You have to fit the solver first by calling the '.fit()' method."
        
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
    from utils import complex_gaussian_matrix
    
    print("Start performing sanity tests...")
    print()
    
    # Variables definition
    cost: int = 1
    sigma: int = 1
    iterations: int = 10
    input_dim: int = 10
    output_dim: int = 2
    antennas_transmitter: int = 4
    antennas_receiver: int = 4
    channel_matrix: torch.Tensor = complex_gaussian_matrix(mean=0, std=1, size=(antennas_receiver, antennas_transmitter))
    
    data = torch.randn(1000, input_dim)
    
    print("Test for Linear Optimizer Baseline...", end='\t')
    baseline = LinearOptimizerBaseline(input_dim=input_dim,
                                       output_dim=output_dim,
                                       channel_matrix=channel_matrix,
                                       sigma=sigma)
    baseline.fit(data, data)
    baseline.transform(data)
    # print(baseline.eval(data, data))
    print("[Passed]")
    
    print()
    print()
    print("Test for Linear Optimizer SAE...", end='\t')
    print()
    linear_sae = LinearOptimizerSAE(input_dim=input_dim,
                                    output_dim=output_dim,
                                    channel_matrix=channel_matrix,
                                    sigma=sigma,
                                    cost=cost)
    linear_sae.fit(data, data, iterations)
    linear_sae.transform(data)
    # print(linear_sae.eval(data, data))
    print("[Passed]")
        
    return None


if __name__ == "__main__":
    main()
