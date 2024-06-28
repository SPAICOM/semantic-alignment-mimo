"""In this module there are defined usefull methods:

- complex_gaussian_matrix:
    A method that returns a complex gaussian matrix in the torch.Tensor format.
"""

import torch


# ================================================================
#
#                        Methods Definition 
#
# ================================================================

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

    x = torch.randn(4, 10)
    # n = torch.normal(mean, std, size=x.shape)
    
    print("Performing first test...", end="\t")
    complex_matrix = complex_gaussian_matrix(mean=mean, std=std, size=size)
    print("[PASSED]")

    print()
    print("Performing second test...", end="\t")
    x = complex_tensor(x)
    print("[PASSED]")

    print()
    print("Performing third test...", end='\t')
    sn_ratio = snr(x.real, std)
    print("[PASSED]")
    
    return None


if __name__ == "__main__":
    main()
