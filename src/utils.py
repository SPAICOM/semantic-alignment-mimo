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

def complex_gaussian_matrix(mean: float,
                            std: float,
                            size: tuple) -> torch.Tensor:
    """
    """
    # Get the real and imaginary parts
    real_part = torch.normal(mean, std/2, size=size)
    imag_part = torch.normal(mean, std/2, size=size)

    # Stack real and imaginary parts along the last dimensioni
    complex_matrix = torch.stack((real_part, imag_part), dim=2)

    return torch.view_as_complex(complex_matrix)
    
    

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

    print("Performing first test...", end="\t")
    complex_matrix = complex_gaussian_matrix(mean=mean, std=std, size=size)
    print("[PASSED]")
    return None


if __name__ == "__main__":
    main()
