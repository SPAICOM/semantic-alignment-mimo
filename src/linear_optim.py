"""This module defines the functions/classes needed for the linear optimization.
"""

import math
import torch
import numpy as np
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
    
from src.utils import complex_tensor, complex_compressed_tensor, decompress_complex_tensor, prewhiten, sigma_given_snr, awgn
# from utils import complex_tensor, complex_compressed_tensor, decompress_complex_tensor, prewhiten, sigma_given_snr, awgn

# ============================================================
#
#                     CLASSES DEFINITION
#
# ============================================================
class LinearOptimizerBaseline():
    """A linear version of the baseline in which we're not taking into account the advantages of semantic communication.

    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension.
        channel_matrix : torch.Tensor
            The channel matrix H in torch.Tensor format.
        snr : float
            The snr in dB of the communication channel. Set to None if unaware.
        typology : str
            The typology of baseline, possible values 'pre' or 'post'. Default 'pre'.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel_matrix: torch.Tensor,
                 snr: float,
                 typology: str = "pre"):
        
        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix
        self.snr = snr
        self.typology = typology

        # Check value of typology
        assert self.typology in ["pre", "post"], f"The passed typology {self.typology} is not supported."        

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
        self.F = Vt.H
        if self.snr:
            self.G = B.H @ torch.linalg.inv(B@B.H + (1/self.snr) * torch.view_as_complex(torch.stack((torch.eye(B.shape[0]), torch.eye(B.shape[0])), dim=-1)))
        else:
            self.G = torch.linalg.inv(S) @ U.H

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
        d = input.shape[0]

        assert d > self.antennas_transmitter, "The self.output dimension must be greater than the number of transmitting antennas"

        # Get the number of packets required for the transmission and the relative needed padding
        n_packets: int = math.ceil( d / self.antennas_transmitter)
        padding_size: int = n_packets * self.antennas_transmitter - d

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
        match self.typology:
            case "pre":
                received = torch.cat(packets, dim=0)[:self.output_dim//2,:]
            case "post":
                received = torch.cat(packets, dim=0)[:self.input_dim//2,:]
            case _:
                raise Exception(f"The passed typology {self.typology} is not supported.")

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
            # Complex Compress the input
            input = complex_compressed_tensor(input.T).H
        
            # Perform the prewhitening step
            input = torch.linalg.solve(self.L, input - self.mean)
        
            # Performing the precoding packets
            packets = self.__packets_precoding(input)

            # Transmit the packets and add the white gaussian noise
            if self.snr:
                # Get the sigma
                packets = [self.channel_matrix @ p + awgn(sigma=sigma_given_snr(snr=self.snr, signal=p), size=p.shape) for p in packets]

            # Decode the packets
            output = self.__packets_decoding(packets)
            
            # Remove whitening
            output = self.L @ output + self.mean

            # Decompress the transmitted signal
            output = decompress_complex_tensor(output.H).T

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

        with torch.no_grad():
            match self.typology:
                case "pre":
                    # Alignment of the input to the output
                    self.A = torch.linalg.lstsq(input, output).solution.T

                    # Align the input
                    input = self.A @ input.T
            
                    # Complex Compress the input
                    input = complex_compressed_tensor(input.T).H

                    # Learn L and the mean
                    _, self.L, self.mean = prewhiten(input)
                
                case "post":
                    # Learn L and the mean
                    _, self.L, self.mean = prewhiten(complex_compressed_tensor(input).H)

                    # Transmit the input
                    transmitted = self.__transmit_message(input.T)

                    # Alignment of the input to the output
                    self.A = torch.linalg.lstsq(transmitted, output).solution.T

                case _:
                    raise Exception(f"The passed typology {self.typology} is not supported.")
            
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

        # Transpose
        input = input.T

        match self.typology:
            case "pre":
                # Align the input
                input = self.A @ input

                # Transmit the input
                output = self.__transmit_message(input)

            case "post":
                # Transmit the input
                output = self.__transmit_message(input)

                # Align the output
                output = (self.A @ output.T).T

            case _:
                raise Exception(f"Unrecognised case of self.typology parameter, set to '{self.typology}'.")

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

    def get_precodings(self,
                       input: torch.Tensor) -> torch.Tensor:
        """The function returns the precodings of a passed input tensor.

        Args:
            input : torch.Tensor
                The input tensor

        Returns:
            precoded : torch.Tensor
                The precoded version of the passed input tensor.
        """
        input.to('cpu')

        # Transpose
        input = input.T

        match self.typology:
            case "pre":
                # Align the input
                input = self.A @ input
                
                # Complex compression
                input = complex_compressed_tensor(input.T).H
                
                # Perform the prewhitening step
                precoded = torch.linalg.solve(self.L, input - self.mean)

            case "post":
                # Complex compression
                input = complex_compressed_tensor(input.T).H
                
                # Perform the prewhitening step
                precoded = torch.linalg.solve(self.L, input - self.mean)
                
            case _:
                raise Exception(f"Unrecognised case of self.typology parameter, set to '{self.typology}'.")

        return precoded.T


    
class LinearOptimizerSAE():
    """The linear optimizer for the Semantic Auto Encoder.
    
    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension.
        channel_matrix : torch.Tensor
            The Complex Gaussian Matrix simulating the communication channel.
        snr : float
            The snr in dB of the communication channel. Set to None if unaware.
        cost : float
            The transmition cost. Default 1.0.
        rho : int
            The rho coeficient for the admm method. Default 1e2.

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
                 snr: float,
                 cost: float = 1.0,
                 rho: float = 1e2):

        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix
        self.snr = snr
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

        # Get the number of samples
        _, n = input.shape

        # Get sigma
        sigma = 0
        if self.snr:
            sigma = sigma_given_snr(self.snr, self.F @ input) / math.sqrt(2)
        
        self.G = output @ A.H @ torch.linalg.inv(A @ A.H + n * sigma * torch.view_as_complex(torch.stack((torch.eye(A.shape[0]), torch.eye(A.shape[0])), dim=-1)))
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
        # Get the number of samples
        _, n = input.shape
        
        # Get the auxiliary matrixes
        rho = self.rho * n
        O = self.G @ self.channel_matrix
        A = O.H @ O
        Bh = (input @ input.H).H
        C = rho * (self.Z - self.U) + O.H @ output @ input.H

        kron = torch.kron(Bh.contiguous(), A.contiguous()) 
        n, m = kron.shape
        
        vec_F = torch.linalg.inv(kron + rho * torch.eye(n, m)) @ C.T.reshape(-1)
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
            
            # Transpose
            input = input.T
            output = output.T

            # Complex compression
            input = complex_compressed_tensor(input.T).H
            output = complex_compressed_tensor(output.T).H

            # Perform the prewhitening step
            input, self.L_input, self.mean_input = prewhiten(input)
            
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
            # Transpose
            input = input.T

            # Complex Compress the input
            input = complex_compressed_tensor(input.T).H
            
            # Perform the prewhitening step
            input = torch.linalg.solve(self.L_input, input - self.mean_input)

            # Transmit the input through the channel H
            z = self.channel_matrix @ self.F @ input

            # Add the additive white gaussian noise
            if self.snr:
                sigma = sigma_given_snr(snr=self.snr, signal= self.F @ input)
                w = awgn(sigma=sigma, size=z.shape)
                z += w
                
            output = self.G @ z
        
            # Decompress the transmitted signal
            output = decompress_complex_tensor(output.H).T
        
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


    def get_precodings(self,
                       input: torch.Tensor) -> torch.Tensor:
        """The function returns the precodings of a passed input tensor.

        Args:
            input : torch.Tensor
                The input tensor

        Returns:
            precoded : torch.Tensor
                The precoded version of the passed input tensor.
        """
        input.to('cpu')

        with torch.no_grad():
            # Transpose
            input = input.T

            # Complex Compress the input
            input = complex_compressed_tensor(input.T).H
            
            # Perform the prewhitening step
            input = torch.linalg.solve(self.L_input, input - self.mean_input)

            # Transmit the input through the channel H
            precoded = self.F @ input

        return precoded


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
    snr: float = 20
    iterations: int = 10
    input_dim: int = 20
    output_dim: int = 40
    antennas_transmitter: int = 4
    antennas_receiver: int = 4
    channel_matrix: torch.Tensor = complex_gaussian_matrix(mean=0, std=1, size=(antennas_receiver, antennas_transmitter))
    
    input = torch.randn(1000, input_dim)
    output = torch.randn(1000, output_dim)
    
    print("Test for Linear Optimizer Baseline...", end='\t')
    baseline = LinearOptimizerBaseline(input_dim=input_dim,
                                       output_dim=output_dim,
                                       channel_matrix=channel_matrix,
                                       snr=snr)
    baseline.fit(input, output)
    baseline.transform(input)
    print(baseline.eval(input, output))
    print("[Passed]")
    
    print()
    print()
    print("Test for Linear Optimizer SAE...", end='\t')
    print()
    linear_sae = LinearOptimizerSAE(input_dim=input_dim,
                                    output_dim=output_dim,
                                    channel_matrix=channel_matrix,
                                    snr=snr,
                                    cost=cost)
    linear_sae.fit(input, output, iterations)
    linear_sae.transform(input)
    print(linear_sae.eval(input, output))
    print("[Passed]")
        
    return None


if __name__ == "__main__":
    main()
