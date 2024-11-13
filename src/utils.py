"""In this module there are defined usefull methods:

- complex_gaussian_matrix:
    A method that returns a complex gaussian matrix in the torch.Tensor format.
"""

import torch
import math


# ================================================================
#
#                        Methods Definition 
#
# ================================================================

def complex_compressed_tensor(x: torch.Tensor) -> torch.Tensor:
    """The function compress the feature dimension of the tensor by converting
    half as real part and the other half as imaginary part.

    Args:
        x : torch.Tensor
            The input tensor to compress.

    Returns:
        torch.Tensor
            The output tensor in complex format.
    """
    n, d = x.shape
    
    if d % 2 != 0:
        x = torch.cat((x, torch.zeros((n, 1), dtype=x.dtype, device=x.device)), dim=1)
        d += 1   # Split the tensor into real and imaginary parts
        
    real_part = x[:, :d//2]
    imaginary_part = x[:, d//2:]

    # Combine real and imaginary parts into a complex tensor
    x = torch.stack((real_part, imaginary_part), dim=-1)

    return torch.view_as_complex(x)


def decompress_complex_tensor(x: torch.Tensor) -> torch.Tensor:
    """The function decompress the complex compressed tensor in the original real domain.

    Args:
        x : torch.Tensor
            The input compressed tensor.

    Returns:
        torch.Tensor
            The output decompressed tensor.
    """
    # Split the complex tensor into real and imaginary parts
    real_part = x.real
    imaginary_part = x.imag

    # Concatenate the real and imaginary parts along the feature dimension
    x = torch.cat((real_part, imaginary_part), dim=1)

    return x


def complex_tensor(x: torch.Tensor) -> torch.Tensor:
    """Get the complex form of a tensor.

    Args:
        x : torch.Tensor
            The original tensor.

    Returns:
        torch.Tensor
            The output tensor, which is the complex form of the original tensor.
    """
    device = x.device
    x = torch.stack((x, torch.zeros(x.shape).to(device)), dim=-1)
    return torch.view_as_complex(x)


def complex_gaussian_matrix(mean: float,
                            std: float,
                            size: tuple[int]) -> torch.Tensor:
    """A method that returns a complex gaussian matrix in the torch.Tensor format.

    Args:
        mean : float
            The mean of the distribution.
        std : float
            The std of the distribution.
        size : tuple[int]
            The size of the matrix.

    Returns:
        torch.Tensor
            The complex gaussian matrix in the torch.Tensor format.
    """
    # Get the real and imaginary parts
    real_part = torch.normal(mean, std/2, size=size)
    imag_part = torch.normal(mean, std/2, size=size)

    # Stack real and imaginary parts along the last dimensioni
    complex_matrix = torch.stack((real_part, imag_part), dim=-1)

    return torch.view_as_complex(complex_matrix)


def snr(signal: torch.Tensor,
        sigma: float) -> float:
    """Return the Signal to Noise Ratio.

    Args:
        signal : torch.Tensor
            The signal vector.
        sigma : float
            The sigma squared of the noise.

    Return:
        float
            The Signal to Noise Ratio.
    """
    return 10*torch.log10(torch.mean(signal**2)/sigma**2).item()


def sigma_given_snr(snr: float,
                    receiving_antennas: int,
                    cost: float = 1.0) -> float:
    """Given a fixed value of SNR, receiving antennas and cost, retrieve the correspoding value of sigma.

    Args:
        snr : float
            The Signal to Noise Ration.
        receiving_antennas : int
            The number of receiving antennas.
        cost : float
            The cost for the transmitter.

    Returns:
        float
            The corresponding sigma given snr, receiving antennas and cost.
    """
    return math.sqrt(cost/(snr*receiving_antennas))


def prewhiten(x_train: torch.Tensor,
              x_test: torch.Tensor = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Prewhiten the training and test data using only training data statistics.
    
    Args:
        x_train : torch.Tensor
            The training torch.Tensor matrix.
        x_test : torch.Tensor
            The testing torch.Tensor matrix. Default None.
            
    Returns:
        z_train : torch.Tensor
            Prewhitened training matrix.
        z_test : torch.Tensor
            Prewhitened test matrix.
    """
    # --- Prewhiten the training set ---
    C = torch.cov(x_train)  # Training set covariance
    L = torch.linalg.cholesky(C)  # Cholesky decomposition C = LL^H
    z_train = x_train - x_train.mean(axis=1)[:,None] # Center the training set
    z_train = torch.linalg.solve(L, z_train)  # Prewhitened training set

    if x_test is not None:
        z_test = x_test - x_train.mean(axis=1)[:,None]  # Center the test set
        z_test = torch.linalg.solve(L, x_test)  # Prewhitened training set
        return z_train, z_test
    
    return z_train

# ================================================================
#
#                        Main Definition 
#
# ================================================================

def main() -> None:
    """Some quality tests...
    """
    
    # Variable definition
    mean: float = 0.
    std: float = 1.
    size: tuple[int] = (4, 4)

    n = 10
    d = 20
    x = torch.randn(n, d)
    # n = torch.normal(mean, std, size=x.shape)
    
    print("Performing first test...", end="\t")
    complex_matrix = complex_gaussian_matrix(mean=mean, std=std, size=size)
    print("[PASSED]")

    print()
    print("Performing second test...", end="\t")
    complex_tensor(x)
    print("[PASSED]")

    print()
    print("Performing third test...", end='\t')
    sn_ratio = snr(x.real, std)
    print("[PASSED]")
    
    print()
    print("Performing fourth test...", end='\t')
    x_c = complex_compressed_tensor(x)
    print("[PASSED]")

    print()
    print("Performing fifth test...", end='\t')
    x_hat = decompress_complex_tensor(x_c)

    if not torch.all(torch.eq(x_hat[:, :d], x)):
        raise Exception("The compression and decompression are not working as intended")
    
    print("[PASSED]")
    
    print()
    print("Performing sixth test...", end="\t")
    prewhiten(x)
    print("[PASSED]")

    print()
    print("Performing seventh test...", end="\t")
    sigma = sigma_given_snr(snr=10, receiving_antennas=100)
    assert sigma > 0, "[Error]: sigma is not positive."
    print("[PASSED]")

    
    return None


if __name__ == "__main__":
    main()
