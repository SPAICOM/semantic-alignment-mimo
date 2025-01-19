"""This module defines the functions/classes needed for the linear optimization.
"""

import math
import torch
import numpy as np
from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from scipy.linalg import solve_sylvester
    
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
        snr_type : str
            The typology of snr, possible values 'transmitted' or 'received'. Default 'transmitted'.
        k_p : int
            Number of packets to consider, if None then all. Default None.
        typology : str
            The typology of baseline, possible values 'pre' or 'post'. Default 'pre'.
        strategy : str
            The strategy to adopt in sending the features, possible values 'first' or 'abs'. Default 'first'.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel_matrix: torch.Tensor,
                 snr: float,
                 snr_type: str = "transmitted",
                 k_p: int = None,
                 typology: str = "post",
                 strategy: str = "first"):
        
        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_matrix = channel_matrix
        self.snr = snr
        self.snr_type = snr_type
        self.k_p = k_p
        self.typology = typology
        self.strategy = strategy
        
        # Check values
        assert self.snr_type in ["transmitted", "received"], f"The passed snr typology {self.snr_type} is not supported."        
        assert self.typology in ["pre", "post"], f"The passed typology {self.typology} is not supported."        
        assert self.strategy in ["first", "abs"], f"The passed strategy {self.strategy} is not supported."        

        # Get the receiver and transmitter antennas
        self.antennas_receiver, self.antennas_transmitter = self.channel_matrix.shape

        # Instantiate the alignment matrix A
        self.A = None 
        
        # Define self.k_p if it set to None
        if not self.k_p:
            self.k_p = math.ceil( (self.input_dim // 2) / self.antennas_transmitter)

        # Perform the SVD of the channel matrix, save the U, S and Vt.
        U, S, Vt = torch.linalg.svd(self.channel_matrix)
        S = torch.diag(S).to(torch.complex64)

        # Auxiliary matrix
        B = U @ S

        # Define the decoder and precoder
        self.F = Vt.H / torch.linalg.norm(Vt.H)
        if self.snr:
            self.G = ( B.H @ torch.linalg.inv(B@B.H + (1/self.snr) * torch.view_as_complex(torch.stack((torch.eye(B.shape[0]), torch.eye(B.shape[0])), dim=-1))) ) * torch.linalg.norm(Vt.H)
        else:
            self.G = ( torch.linalg.inv(S) @ U.H ) * torch.linalg.norm(Vt.H)

        return None
    

    def __compression(self,
                      input: torch.Tensor) -> torch.Tensor:
        """Compress the input.

        Args:
            input : torch.Tensor
                The input as real d x n.

        Return:
            input : torch.Tensor
                The compressed input as complex k_p * N_t x n.
        """
        # Get the number of features of the input
        self.size, n = input.shape

        # Features to transmit
        self.sent_features = 2 * self.k_p * self.antennas_transmitter 
        
        if self.strategy == "first":
            input = input[:self.sent_features, :]

        elif self.strategy == "abs":
            # Get the indexes based on the selected strategy
            _, self.indexes = torch.topk(input.abs(), self.sent_features, dim=0)

            # Retrieve the values based on the indexes
            input = input[self.indexes, torch.arange(n)]

        else:
            raise Exception("The passed strategy is not supported.")

        # Complex Compression
        input = complex_compressed_tensor(input.T).H

        return input


    def __decompression(self,
                        received: torch.Tensor) -> torch.Tensor:
        """Decompression of the received message.

        Args:
            received : torch.Tensor
                The received message

        Return:
            output : torch.Tensor
                The output.
        """
        _, n = received.shape
        
        # Decompress the transmitted signal
        received = decompress_complex_tensor(received.H).T

        output = torch.zeros(self.size, n)         
        
        if self.strategy == "first":
            output[:self.sent_features, :] = received

        elif self.strategy == "abs":
            output[self.indexes, torch.arange(n)] = received
            
        else:
            raise Exception("The passed strategy is not supported.")
        
        return output   
    

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
        # Compress the input
        input = self.__compression(input)        
        
        # Perform the prewhitening step
        input = torch.linalg.solve(self.L, input - self.mean)
        
        # Create the packets of size self.antennas_transmitter
        packets = torch.split(input, self.antennas_transmitter, dim=0)
        
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
                The output seen as a single torch.Tensor of dimension self.output_dim x num. of observation.
        """
        # Decode the packets
        packets = [self.G @ p for p in packets]

        # Concat the packets
        received = torch.cat(packets, dim=0)

        # Remove whitening
        received = self.L @ received + self.mean
        
        # Decompress the transmitted signal
        received = self.__decompression(received)

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
            # Performing the precoding packets
            packets = self.__packets_precoding(input)


            # Transmit and add the AWGN if needed
            if self.snr:
                # Get the sigma
                if self.snr_type == "received":
                    # Transmit the packets
                    packets = [self.channel_matrix @ p for p in packets]
                    packets = [p + awgn(sigma=sigma_given_snr(snr=self.snr, signal=p), size=p.shape) for p in packets]
                elif self.snr_type == "transmitted":
                    # Transmit the packets
                    # packets = [self.channel_matrix @ p + awgn(sigma=sigma_given_snr(snr=self.snr, signal=p), size=p.shape) for p in packets]
                    sigma = sigma_given_snr(snr=self.snr, signal=torch.ones(1)/math.sqrt(self.antennas_transmitter))
                    packets = [self.channel_matrix @ p + awgn(sigma=sigma, size=p.shape) for p in packets]
                else:
                    raise Exception("Wrong snr typology passed.")
            else:
                # Transmit the packets
                packets = [self.channel_matrix @ p for p in packets]

            # Decode the packets
            output = self.__packets_decoding(packets)
            
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
            # Alignment of the input to the output
            self.A = torch.linalg.lstsq(input, output).solution.T
                    
            match self.typology:
                case "pre":
                    # Align the input
                    input = self.A @ input.T
            
                    # Compress the input
                    input = self.__compression(input)

                    # Learn L and the mean
                    _, self.L, self.mean = prewhiten(input)
                
                case "post":
                    # Compress the input
                    input = self.__compression(input)

                    # Learn L and the mean
                    _, self.L, self.mean = prewhiten(input)

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
        snr_type : str
            The typology of snr, possible values 'transmitted' or 'received'. Default 'transmitted'.
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
                 snr_type: str = "transmitted",
                 cost: float = 1.0,
                 rho: float = 1e2,
                 device: str = "cpu"):

        assert len(channel_matrix.shape) == 2, "The matrix must be 2 dimesional."
        
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.channel_matrix: torch.Tensor = channel_matrix.to(device)
        self.snr: float = snr
        self.snr_type: str = snr_type
        self.cost: float = cost
        self.rho: float = rho
        self.device: str = device
        self.dtype = channel_matrix.dtype
        self.antennas_receiver, self.antennas_transmitter = self.channel_matrix.shape
        
        # Check values
        assert self.snr_type in ["transmitted", "received"], f"The passed snr typology {self.snr_type} is not supported."        

        # Variables
        self.F = None
        self.G = None

        # ADMM variables
        self.Z = torch.zeros(self.antennas_transmitter, (self.input_dim + 1) // 2).to(self.device)
        self.U = torch.zeros(self.antennas_transmitter, (self.input_dim + 1) // 2).to(self.device)

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
            if self.snr_type == "transmitted":
                # sigma = sigma_given_snr(self.snr, self.F @ input)
                sigma = sigma_given_snr(self.snr, torch.ones(1)/math.sqrt(self.antennas_transmitter))
            elif self.snr_type == "received":
                sigma = sigma_given_snr(self.snr, A)
            else:
                raise Exception("Wrong snr typology passed.")
        
        self.G = (output @ A.H @ torch.linalg.inv(A @ A.H + n * sigma * torch.view_as_complex(torch.stack((torch.eye(A.shape[0]), torch.eye(A.shape[0])), dim=-1)).to(self.device))).to(self.device)
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
        B = rho * torch.linalg.inv(input @ input.H)
        C = (rho * (self.Z - self.U) + O.H @ output @ input.H) @ (B/rho)

        self.F = torch.tensor(solve_sylvester(A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy()), device=self.device, dtype=self.dtype)
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
        with torch.no_grad():
            # Inizialize the F matrix at random
            self.F = torch.view_as_complex(torch.stack((torch.randn(self.antennas_transmitter, (self.input_dim + 1) // 2), torch.randn(self.antennas_transmitter, (self.input_dim + 1) // 2)), dim=-1)).to(self.device)

            # Save the decompressed version
            old_input = input
            old_output = output
            
            # Transpose
            input = input.T
            output = output.T

            # Complex compression
            input = complex_compressed_tensor(input.T, device=self.device).H
            output = complex_compressed_tensor(output.T, device=self.device).H

            # Perform the prewhitening step
            input, self.L_input, self.mean_input = prewhiten(input, device=self.device)
            
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
        with torch.no_grad():
            # Transpose
            input = input.T

            # Complex Compress the input
            input = complex_compressed_tensor(input.T, device=self.device).H
            
            # Perform the prewhitening step
            input = torch.linalg.solve(self.L_input, input - self.mean_input)

            # Transmit the input through the channel H
            z = self.channel_matrix @ self.F @ input

            # Add the additive white gaussian noise
            if self.snr:
                if self.snr_type == "transmitted":
                    # sigma = sigma_given_snr(snr=self.snr, signal= self.F @ input)
                    sigma = sigma_given_snr(snr=self.snr, signal=torch.ones(1)/math.sqrt(self.antennas_transmitter))
                elif self.snr_type == "received":
                    sigma = sigma_given_snr(snr=self.snr, signal= z)
                else:
                    raise Exception("Wrong snr typology passed.")
                
                w = awgn(sigma=sigma, size=z.shape, device=self.device)
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
    snr: float = 20
    snr_type: str = "received"
    k_p: int = 1
    iterations: int = 10
    input_dim: int = 384
    output_dim: int = 768
    antennas_transmitter: int = 4
    antennas_receiver: int = 4
    channel_matrix: torch.Tensor = complex_gaussian_matrix(mean=0, std=1, size=(antennas_receiver, antennas_transmitter))
    
    input = torch.randn(100, input_dim)
    output = torch.randn(100, output_dim)
    
    print("Test for Linear Optimizer Baseline...", end='\t')
    baseline = LinearOptimizerBaseline(input_dim=input_dim,
                                       output_dim=output_dim,
                                       channel_matrix=channel_matrix,
                                       snr=snr,
                                       snr_type=snr_type,
                                       k_p=k_p,
                                       strategy="first")
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
                                    snr_type=snr_type,
                                    cost=cost,
                                    device="cpu")
    linear_sae.fit(input, output, iterations)
    linear_sae.transform(input)
    print(linear_sae.eval(input, output))
    print("[Passed]")
        
    return None


if __name__ == "__main__":
    main()
