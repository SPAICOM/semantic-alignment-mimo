"""A module in which the are definined the similiarities functions.
"""

import torch
import torch.nn.functional as F


# ================================================================
#
#                        METHODS DEFINITION
#
# ================================================================

def relative_projection(x: torch.Tensor,
                        anchors: torch.Tensor) -> torch.Tensor:
    """Given a set of anchors and of x observations, it returns the projection of x over the anchors.

    Args:
        - x (torch.Tensor): The observations x.
        - anchors (torch.Tensor): The anchors.

    Returns:
        - torch.Tensor : The projection of x over the anchors.
    """
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1)
    return torch.einsum("bm, am -> ba", x, anchors)



# ================================================================
#
#                        MAIN DEFINITION
#
# ================================================================

def main() -> None:
    """Some quality checks...
    """
    print("Start performing sanity tests...")
    print()

    x: torch.Tensor = torch.randn(5, 10)
    anchors: torch.Tensor = torch.randn(2, 10)

    print("First test...", end='\t')
    relative_projection(x, anchors)
    print("[PASSED]")
    
    return None


if __name__ == "__main__":
    main()
