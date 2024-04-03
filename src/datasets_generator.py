"""
This python module creates syntetic datasets for the task.

To check the available parameters just run 'python /path/to/datasets_generator.py -h'.
"""

import numpy as np
import polars as pl
from pathlib import Path
from numpy.linalg import norm

def main() -> None:
    """The main script loop.
    """
    import argparse

    description = """
    This python module creates syntetic datasets for the task.

    To check the available parameters just run 'python /path/to/datasets_generator.py -h'.
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-d',
                        '--dimension',
                        help='The vectors dimension. Default 10.',
                        default=10,
                        type=int)

    parser.add_argument('-o',
                        '--observations',
                        help="The number of observations (vectors) to generate. Default 1'000.",
                        default=1000,
                        type=int)

    parser.add_argument('-a',
                        '--anchors',
                        help='The number of anchors. Default 10.',
                        default=10,
                        type=int)

    parser.add_argument('-f',
                        '--function',
                        help="The similarity/distance function type that is use for building the relative rappresentation. Default 'tanh'.",
                        choices=['tanh', 'sigmoid', 'softplus'],
                        default='tanh',
                        type=str)

    args = parser.parse_args()
    
    # Directory paths
    current = Path('.')
    data_path = current / 'data'

    # Sanity check
    assert data_path.is_dir(), f"The passed path './{data_path}/' doesn't exists."

    # Attributes definition
    maximum = 10
    rel_type = args.function
    dim = args.dimension
    n_obs = args.observations
    n_anchors = args.anchors

    # Generate anchors and observations
    anchors = np.random.uniform(-maximum, maximum, size=(n_anchors, dim))
    obs = np.random.uniform(-maximum, maximum, size=(n_obs, dim))

    # Manipulation need to create a DataFrame
    anchors = np.tile(anchors, (n_obs, 1))
    obs = np.repeat(obs, n_anchors, axis=0)

    # Relative score for fun(x_i, a_j)
    if rel_type == "tanh":
        rel_ij = np.sum(obs*anchors, axis=1)/(norm(obs, axis=1)*norm(anchors, axis=1))
    elif rel_type == "sigmoid":
        rel_ij = (1 + np.sum(obs*anchors, axis=1)/(norm(obs, axis=1)*norm(anchors, axis=1)))/2
    elif rel_type == "softplus":
        rel_ij = norm(obs-anchors, axis=1)

    # Create and save the DataFrame
    pl.DataFrame({'x_i': obs,
                  'a_j': anchors,
                  'r_ij': rel_ij}).write_parquet(data_path/f'example_n{n_obs}_dim{dim}_{rel_type}_anchors{n_anchors}.parquet')


if __name__ == "__main__":
    main()
