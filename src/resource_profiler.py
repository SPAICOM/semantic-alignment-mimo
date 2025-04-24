"""Module containg needed methods to compute model complexity."""

import torch
import polars as pl
import seaborn as sns
from pathlib import Path
from math import ceil, floor
import matplotlib.pyplot as plt


# ==================================================================
#
#                       FUNCTIONS DEFINITION
#
# ==================================================================


def linear_flops(
    transmitter: int,
    receiver: int,
    input_dim: int = 192,
    output_dim: int = 384,
) -> int:
    """Calculates the flops required for the linear case.

    Args:
        transmitter : int
            The number of antennas in the transmitter.
        receiver : int
            The number of antennas in the receiver.
        input_dim : int
            The input dimension. Default 192.
        output_dim : int
            The output dimension. Default 384.

    Returns:
        int:
            The total number of flops required.
    """
    encoder = transmitter * (8 * input_dim - 2)
    decoder = output_dim * (8 * receiver - 2)

    print('Flops semantic encoder: ', encoder)
    print('Flops semantic decoder: ', decoder)
    return encoder + decoder


def neural_flops(
    transmitter: int,
    receiver: int,
    input_dim: int = 192,
    output_dim: int = 384,
    enc_hidden: int = 192,
    dec_hidden: int = 384,
    hidden_size: int = 0,
) -> int:
    """Calculates the flops required for the neural case.

    Args:
        transmitter : int
            The number of antennas in the transmitter.
        receiver : int
            The number of antennas in the receiver.
        input_dim : int
            The input dimension. Default 192.
        output_dim : int
            The output dimension. Default 384.
        enc_hidden : int
            The hidden dimension of the encoder. Default 192.
        dec_hidden : int
            The hidden dimension of the decoder. Default 384.
        hidden_size : int
            How many hidden layers for both encoder and decoder. Default 0.

    Returns:
        int:
            The total number of flops required.
    """
    cgelu = 100

    input_layer = enc_hidden * (8 * input_dim) + cgelu * enc_hidden
    hidden_layers = (
        enc_hidden * (8 * enc_hidden) + cgelu * enc_hidden
    ) * hidden_size
    output_layer = transmitter * (8 * enc_hidden)
    encoder = floor(input_layer + hidden_layers + output_layer)

    input_layer = dec_hidden * (8 * receiver) + cgelu * dec_hidden
    hidden_layers = (
        dec_hidden * (8 * dec_hidden) + cgelu * dec_hidden
    ) * hidden_size
    output_layer = output_dim * (8 * dec_hidden)
    decoder = ceil(input_layer + hidden_layers + output_layer)

    print('Flops semantic encoder: ', encoder)
    print('Flops semantic decoder: ', decoder)
    return encoder + decoder


def neural_sparse_flops(
    transmitter: int,
    receiver: int,
    sparsity: float,
    input_dim: int = 192,
    output_dim: int = 384,
    enc_hidden: int = 192,
    dec_hidden: int = 384,
    hidden_size: int = 0,
) -> int:
    """Calculare the flops required for the neural sparse case.

    Args:
        transmitter : int
            The number of antennas in the transmitter.
        receiver : int
            The number of antennas in the receiver.
        sparsity : float
            The sparsity level.
        input_dim : int
            The input dimension. Default 192.
        output_dim : int
            The output dimension. Default 384.
        enc_hidden : int
            The hidden dimension of the encoder. Default 192.
        dec_hidden : int
            The hidden dimension of the decoder. Default 384.
        hidden_size : int
            How many hidden layers for both encoder and decoder. Default 0.

    Returns:
        int:
            The total number of flops required.
    """
    density = 1 - sparsity

    cgelu = 100

    input_layer = enc_hidden * (8 * input_dim * density) + cgelu * enc_hidden
    hidden_layers = (
        enc_hidden * (8 * enc_hidden * density) + cgelu * enc_hidden
    ) * hidden_size
    output_layer = transmitter * (8 * enc_hidden * density)
    encoder = floor(input_layer + hidden_layers + output_layer)

    input_layer = dec_hidden * (8 * receiver * density) + cgelu * dec_hidden
    hidden_layers = (
        dec_hidden * (8 * dec_hidden * density) + cgelu * dec_hidden
    ) * hidden_size
    output_layer = output_dim * (8 * dec_hidden * density)
    decoder = ceil(input_layer + hidden_layers + output_layer)

    print('Flops semantic encoder: ', encoder)
    print('Flops semantic decoder: ', decoder)
    return encoder + decoder


def count_nonzero_weights(model: torch.nn.Module) -> tuple[int, int]:
    """Calculates the amount of nonzero weights in a pytorch model.

    Args:
        model : torch.nn.Module
            The neural pytorch model.
    Returns:
        nonzero, total : tuple[int, int]
            The non zero amount of weights and their total number.
    """
    nonzero = 0
    total = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            nonzero += torch.count_nonzero(param).item()
            total += param.numel()
    return nonzero, total


def main() -> None:
    """The main loop."""
    from src.neural_models import NeuralModel

    CURRENT: Path = Path('.')
    MODELS: Path = CURRENT / 'models'
    IMG_PATH: Path = CURRENT / 'img'

    device = 'cuda'
    checkpoints = 'autoencoders_pruned'
    input_dim: int = 192
    output_dim: int = 384

    results = pl.DataFrame(
        schema=[
            ('Dataset', pl.String),
            ('Encoder', pl.String),
            ('Decoder', pl.String),
            ('Transmitting Antennas', pl.Int64),
            ('Receiving Antennas', pl.Int64),
            ('SNR', pl.Float64),
            ('Seed', pl.Int64),
            ('SAE', pl.Int64),
            ('SAE Sparse', pl.Int64),
            ('Linear', pl.Int64),
            ('Sparsity', pl.Float64),
            ('Neural Semantic Precoding/Decoding', pl.Int64),
            ('Neural Semantic Sparse Precoding/Decoding', pl.Int64),
            ('Linear Semantic Precoding/Decoding', pl.Int64),
        ]
    )

    for ckpt_path in (MODELS / f'{checkpoints}/').rglob('*.ckpt'):
        print()
        print()
        # Getting the settings
        _, _, dataset, encoder, decoder, awareness, antennas, snr, seed = str(
            ckpt_path.as_posix()
        ).split('/')
        transmitter, receiver = list(map(int, antennas.split('_')[-2:]))
        snr = float(snr.split('_')[-1])
        seed = int(seed.split('.')[0].split('_')[-1])

        if awareness != 'aware':
            continue

        print('#' * 100)
        print(ckpt_path)

        model = NeuralModel.load_from_checkpoint(ckpt_path).to(device)

        model.example_input_array.to(device)

        nonzero, tot = count_nonzero_weights(model)

        sparsity = 1 - nonzero / tot

        results.vstack(
            pl.DataFrame(
                {
                    'Dataset': dataset,
                    'Encoder': encoder,
                    'Decoder': decoder,
                    'Transmitting Antennas': transmitter,
                    'Receiving Antennas': receiver,
                    'SNR': snr,
                    'Seed': seed,
                    'SAE': tot,
                    'SAE Sparse': nonzero,
                    'Linear': transmitter * input_dim + output_dim * receiver,
                    'Sparsity': sparsity,
                    'Neural Semantic Precoding/Decoding': neural_flops(
                        transmitter, receiver, input_dim, output_dim
                    ),
                    'Neural Semantic Sparse Precoding/Decoding': neural_sparse_flops(
                        transmitter, receiver, sparsity, input_dim, output_dim
                    ),
                    'Linear Semantic Precoding/Decoding': linear_flops(
                        transmitter, receiver, input_dim, output_dim
                    ),
                }
            ),
            in_place=True,
        )

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

        results.write_parquet('flops.parquet')
    print(results)

    filter = (pl.col('Awareness') == 'aware') & (pl.col('SNR') == 20)

    results = (
        results.unpivot(
            on=[
                'Neural Semantic Precoding/Decoding',
                'Linear Semantic Precoding/Decoding',
                'Neural Semantic Sparse Precoding/Decoding',
            ],
            index=['Transmitting Antennas', 'Seed'],
            variable_name='Case',
            value_name='FLOPs',
        )
        .join(
            pl.read_parquet('final_results.parquet')
            .vstack(
                pl.read_parquet('final_results_pruned.parquet').with_columns(
                    pl.when(
                        pl.col('Case') == 'Neural Semantic Precoding/Decoding'
                    )
                    .then(pl.lit('Neural Semantic Sparse Precoding/Decoding'))
                    .otherwise(pl.col('Case'))
                    .alias('Case')
                )
            )
            .filter(filter)
            .select(['Accuracy', 'Transmitting Antennas', 'Case', 'Seed']),
            on=['Transmitting Antennas', 'Case', 'Seed'],
            how='left',
        )
        .with_columns(
            ((pl.col('Transmitting Antennas') / input_dim) * 100)
            .round(2)
            .alias('Semantic Compression Factor %')
        )
        .sort(by='FLOPs', descending=False)
    )

    print(results)

    sns.barplot(
        results.to_pandas(),
        x='Semantic Compression Factor %',
        y='FLOPs',
        hue='Case',
    ).set(yscale='log')

    plt.savefig(str(IMG_PATH / 'flops.pdf'), format='pdf')
    plt.show()

    sns.lineplot(
        results.filter(pl.col('Seed') == 200).to_pandas(),
        x='FLOPs',
        y='Accuracy',
        hue='Case',
    ).set(xscale='log')

    plt.savefig(str(IMG_PATH / 'accuracy_vs_complexity.pdf'), format='pdf')
    plt.show()

    return None


if __name__ == '__main__':
    main()
