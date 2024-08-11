"""
"""
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import torch
import polars as pl
from math import ceil, floor
# from calflops import calculate_flops
# from torch.profiler import profile, record_function, ProfilerActivity
from src.models import SemanticAutoEncoder


def linear_flops(transmitter: int,
                 receiver: int,
                 input_dim: int = 192,
                 output_dim: int = 384) -> int:
    """
    """
    encoder = transmitter * (8*input_dim - 2)
    decoder = output_dim * (8*receiver-2)

    print("Flops semantic encoder: ", encoder)
    print("Flops semantic decoder: ", decoder)
    return encoder + decoder

def neural_flops(transmitter: int,
                 receiver: int,
                 input_dim: int = 192,
                 output_dim: int = 384,
                 enc_hidden: int = 192,
                 dec_hidden: int = 384,
                 hidden_size: int = 0) -> int:
    """
    """
    cgelu = 100
    
    input_layer = enc_hidden * (8 * input_dim)
    act = cgelu * input_dim
    hidden_layers = (enc_hidden * (8 * enc_hidden)  + cgelu * enc_hidden) * hidden_size
    output_layer = transmitter * (8 * enc_hidden)
    encoder = floor(input_layer + act + hidden_layers + output_layer)

    input_layer = dec_hidden * (8 * receiver)
    act = cgelu * receiver
    hidden_layers = (dec_hidden * (8 * dec_hidden)  + cgelu * dec_hidden) * hidden_size
    output_layer = output_dim * (8 * dec_hidden)
    decoder = ceil(input_layer + act + hidden_layers + output_layer)

    print("Flops semantic encoder: ", encoder)
    print("Flops semantic decoder: ", decoder)
    return encoder + decoder


def neural_sparse_flops(transmitter: int,
                        receiver: int,
                        sparsity: float,
                        input_dim: int = 192,
                        output_dim: int = 384,
                        enc_hidden: int = 192,
                        dec_hidden: int = 384,
                        hidden_size: int = 0) -> int:
    """
    """
    density = 1-sparsity
    
    cgelu = 100
    
    input_layer = enc_hidden * (8 * input_dim * density)
    act = cgelu * input_dim
    hidden_layers = (enc_hidden * (8 * enc_hidden * density)  + cgelu * enc_hidden) * hidden_size
    output_layer = transmitter * (8 * enc_hidden * density)
    encoder = floor(input_layer + act + hidden_layers + output_layer)

    input_layer = dec_hidden * (8 * receiver * density)
    act = cgelu * receiver
    hidden_layers = (dec_hidden * (8 * dec_hidden * density)  + cgelu * dec_hidden) * hidden_size
    output_layer = output_dim * (8 * dec_hidden * density)
    decoder = ceil(input_layer + act + hidden_layers + output_layer)

    print("Flops semantic encoder: ", encoder)
    print("Flops semantic decoder: ", decoder)
    return encoder + decoder


def count_nonzero_weights(model) -> tuple[int, int]:
    nonzero = 0
    total = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            nonzero += torch.count_nonzero(param).item()
            total += param.numel()
    return nonzero, total


def main() -> None:
    """
    """
    CURRENT: Path = Path('.')
    MODELS: Path = CURRENT / 'models'

    device = 'cuda'
    checkpoints = "autoencoders_sparse"
    target = "abs"
    input_dim: int = 192
    output_dim: int = 384
    
    results = pl.DataFrame(schema=[
                               ('Dataset', pl.String),
                               ('Encoder', pl.String),
                               ('Decoder', pl.String),
                               ('Transmitting Antennas', pl.Int64),
                               ('Receiving Antennas', pl.Int64),
                               ('Awareness', pl.String),
                               ('Sigma', pl.Float64),
                               ('Seed', pl.Int64),
                               ('SAE', pl.Int64),
                               ('SAE Sparse', pl.Int64),
                               ('Linear', pl.Int64),
                               ('Sparsity', pl.Float64),
                               ('SAE FLOPs', pl.Int64),
                               ('SAE Sparse FLOPs', pl.Int64),
                               ('Linear FLOPs', pl.Int64),
                           ])
    
    for ckpt_path in (MODELS / f'{checkpoints}/{target}').rglob('*.ckpt'):
        print()
        print()
        print('#'*100)
        print(ckpt_path)
        # Getting the settings
        _, _, _, dataset, encoder, decoder, case, awareness, antennas, sigma, seed = str(ckpt_path.as_posix()).split('/')
        transmitter, receiver = list(map(int, antennas.split('_')[-2:]))
        case = case.split('_')[-1]
        sigma = float(sigma.split('_')[-1])
        seed = int(seed.split('.')[0].split('_')[-1])

    
        model = SemanticAutoEncoder.load_from_checkpoint(ckpt_path).to(device)

        input = model.example_input_array.to(device)
        
        nonzero, tot = count_nonzero_weights(model)

        sparsity = 1 - nonzero/tot

        print(nonzero, tot, sparsity)
    
        print(neural_sparse_flops(transmitter, receiver, 0.9, input_dim, output_dim))
        
        results.vstack(pl.DataFrame(
                       {
                           'Dataset': dataset,
                           'Encoder': encoder,
                           'Decoder': decoder,
                           'Transmitting Antennas': transmitter,
                           'Receiving Antennas': receiver,
                           'Awareness': awareness,
                           'Sigma': sigma,
                           'Seed': seed,
                           'SAE': tot,
                           'SAE Sparse': nonzero,
                           'Linear': transmitter*input_dim + output_dim*receiver,
                           'Sparsity': sparsity,
                           'SAE FLOPs':  neural_flops(transmitter, receiver, input_dim, output_dim),
                           'SAE Sparse FLOPs': neural_sparse_flops(transmitter, receiver, sparsity, input_dim, output_dim),
                           'Linear FLOPs': linear_flops(transmitter, receiver, input_dim, output_dim),
                       }),
                       in_place=True)

        
               
        # flops, macs, params = calculate_flops(model=model,
        #                                       input_shape=tuple(input.shape),
        #                                       output_as_string=False,
        #                                       output_precision=4)

        # print(f"{flops=}, {macs=}, {params=}")

    
        # activities=[
        #     torch.profiler.ProfilerActivity.CPU,
        #     torch.profiler.ProfilerActivity.CUDA
        # ]

        # with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
        #     with record_function("model_inference"):
        #         model(input)

        # print(prof.key_averages().table(sort_by="cpu_time_total"))
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total"))
        # prof.export_chrome_trace("trace.json")

    
        results.write_parquet(f'flops.parquet')
    print(results)
    
    return None

if __name__ == "__main__":
    main()
