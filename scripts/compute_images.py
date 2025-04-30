"""A usefull script used to compute the needed plots.

The script expects to have the results saved in the following structure (both linear and baseline results):
|_ results/
    |_ neural/
    |   |_ r1.parquet
    |   |_ ...
    |   |_ rk.parquet
    |_ linear/
    |   |_ r1.parquet
    |   |_ ...
    |   |_ rk.parquet
    |_ baseline/
        |_ r1.parquet
        |_ ...
        |_ rk.parquet
"""

import polars as pl
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================


def main() -> None:
    """The main loop."""
    # Defining some usefull paths
    CURRENT: Path = Path('.')
    RESULTS_PATH: Path = CURRENT / 'results/'
    IMG_PATH: Path = CURRENT / 'img'

    # Create image Path
    IMG_PATH.mkdir(exist_ok=True)

    # Set sns style
    sns.set_style('whitegrid')

    # Set style
    plt.style.use('.conf/plotting/plt.mplstyle')

    # Retrieve Data
    df: pl.DataFrame = (
        pl.read_parquet(RESULTS_PATH / 'neural/*.parquet')
        .vstack(pl.read_parquet(RESULTS_PATH / 'linear/*.parquet'))
        .vstack(pl.read_parquet(RESULTS_PATH / 'baseline/*.parquet'))
        .rename(
            {
                'Antennas Transmitter': 'Channel',
                'Training Label Size': 'Semantic Pilots per Class',
            }
        )
    )

    df = df.with_columns(
        (
            (
                pl.col('Channel')
                / df.filter(pl.col('Case').str.starts_with('Neural'))[
                    'Latent Complex Dim'
                ].max()
            )
            * 100
        )
        .round(2)
        .alias('Compression Factor'),
        (
            (pl.col('Channel')).cast(pl.String)
            + 'x'
            + (pl.col('Channel')).cast(pl.String)
        ).alias('Channel'),
    ).sort('Awareness')

    # ===================================================================================
    #                          Accuracy Vs Compression Factor
    # ===================================================================================
    filter = pl.col('Simulation') == 'compr_fact'

    # Define ticks
    ticks = list(
        map(int, df.filter(filter)['Compression Factor'].unique().to_list())
    )
    ticks.remove(0)
    ticks.append(df.filter(filter)['Compression Factor'].min())

    ax = sns.lineplot(
        df.filter(filter),
        x='Compression Factor',
        y='Accuracy',
        style='Semantic Pilots per Class',
        hue='Case',
        markers=True,
    )
    # Get all handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Separate by Case and Channel
    case_labels = [
        'Neural Semantic Precoding/Decoding',
        'Linear Semantic Precoding/Decoding',
        'Baseline Eigen-K',
        'Neural Semantic Precoding/Decoding - Channel Unaware',
        'Linear Semantic Precoding/Decoding - Channel Unaware',
    ]
    style_labels = (
        df.filter(filter)['Semantic Pilots per Class']
        .sort(descending=False)
        .unique()
        .to_list()
    )

    # Match labels to handles
    case_handles = [handles[labels.index(cl)] for cl in case_labels]
    style_handles = [handles[labels.index(str(cl))] for cl in style_labels]

    # First legend: Style
    legend1 = ax.legend(
        style_handles,
        style_labels,
        title='Semantic Pilots per Class',
        loc='upper right',
        bbox_to_anchor=(1, 0.7),
        ncol=3,
        frameon=True,
        framealpha=1,
    )

    # Second legend: Case
    ax.legend(
        case_handles,
        case_labels,
        title='Case',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        frameon=True,
    )

    ax.add_artist(legend1)

    plt.xscale('log')
    plt.xlabel(r'Compression Factor $\zeta$ (\%)')
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.savefig(
        str(IMG_PATH / 'AccuracyVsCompression_factor.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(IMG_PATH / 'AccuracyVsCompression_factor.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    # ===================================================================================
    #                          Accuracy Vs Signal to Noise Ratio
    # ===================================================================================
    filter = pl.col('Simulation') == 'snr'

    ticks = df.filter(filter)['SNR'].unique().to_list()

    hue_order = [
        'Neural Semantic Precoding/Decoding',
        'Linear Semantic Precoding/Decoding',
        'Baseline Eigen-K',
        'Baseline Top-K',
        'Baseline First-K',
    ]

    ax = sns.lineplot(
        df.filter(filter),
        x='SNR',
        y='Accuracy',
        hue='Case',
        style='Case',
        markers=True,
        dashes=False,
        hue_order=hue_order,
    )
    sns.move_legend(
        ax,
        'upper center',
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.2),
    )
    plt.xlabel('Signal to Noise Ratio (dB)')
    plt.xlim(min(ticks), max(ticks))
    plt.savefig(
        str(IMG_PATH / 'AccuracyVsSNR.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(IMG_PATH / 'AccuracyVsSNR.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    # ===================================================================================
    #                          Accuracy Vs Signal to Noise Ratio
    # ===================================================================================
    filter = pl.col('Simulation') == 'pgd'

    plot_df = (
        df.filter(filter)
        .group_by(['Compression Factor', 'Case', 'Lambda'])
        .agg(
            pl.col('FLOPs').mean(),
            pl.col('Accuracy').mean(),
        )
        .with_columns(
            Case=pl.when(
                pl.col('Case') == 'Neural Semantic Precoding/Decoding'
            )
            .then(
                pl.lit(r'Neural Semantic $\zeta =')
                + pl.col('Compression Factor').cast(pl.Int64).cast(pl.String)
                + pl.lit(r'\%$ PDG with Hard Thresholding')
            )
            .when(pl.col('Case') == 'Linear Semantic Precoding/Decoding')
            .then(
                pl.lit(r'Linear Semantic $\zeta =')
                + pl.col('Compression Factor').cast(pl.Int64).cast(pl.String)
                + pl.lit(r'\%$')
            )
        )
        .sort(['Compression Factor', 'Case'], descending=True)
    )

    sns.lineplot(
        plot_df,
        x='FLOPs',
        y='Accuracy',
        hue='Case',
        style='Case',
        dashes=False,
        markers=True,
    ).set(ylim=(0, 1))
    plt.xlabel('FLOPs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(
        str(IMG_PATH / 'AccuracyVsFLOPs.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(IMG_PATH / 'AccuracyVsFLOPs.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    return None


if __name__ == '__main__':
    main()
