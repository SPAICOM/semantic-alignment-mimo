"""This python module handles the final visualizations.

    To check available parameters run 'python /path/to/get_images.py --help'.
"""

# Add root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent))

import polars as pl
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

def main() -> None:
    """The main loop.
    """
    import argparse

    description = """
    This python module handles the final visualizations.

    To check available parameters run 'python /path/to/get_images.py --help'.
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-f',
                        '--file',
                        help="The parquet file with the results.",
                        type=str,
                        default="results.parquet")

    parser.add_argument('-t',
                        '--type',
                        help="The snr type.",
                        type=str,
                        choices=["transmitted", "received"],
                        default="transmitted")

    args = parser.parse_args()

    # Define some paths
    CURRENT: Path = Path('.')
    PARQUET_PATH: Path = CURRENT / args.file
    IMG_PATH: Path = CURRENT / 'img'

    # Create the 'img' folder if it doesn't exist
    IMG_PATH.mkdir(parents=True, exist_ok=True)

    # Set the plot style in seaborn
    sns.set_style('whitegrid')
    
    # Read the parquet
    df = pl.read_parquet(PARQUET_PATH).sort(by=["Awareness", "Case"], descending=[False, True]).with_columns(pl.when(pl.col("Case").str.contains("Baseline")).then((5, 5)).otherwise(()).alias("dashes"))
    dashes = df.select(["Case", "dashes"]).unique(subset=["Case"]).to_dict(as_series= False)
    dashes = dict(zip(dashes["Case"], dashes["dashes"]))

    # Set the font to Times-Roman (or Times New Roman)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times']

    plt.rcParams.update({
                        'font.size': 22,  # General font size
                        'axes.titlesize': 24,  # Title font size
                        'axes.labelsize': 24,  # X and Y label size
                        'xtick.labelsize': 22,  # X-tick label size
                        'ytick.labelsize': 22,  # Y-tick label size
                        'legend.fontsize': 20,  # Legend font size
                        'legend.title_fontsize': 22,  # Legend title font size
                        'text.usetex': True
                        })
    markersize = 18
    linewidth = 3
    
    # ====================================================================================================================
    #                                        Antennas Absolute
    # ====================================================================================================================
    filter = (pl.col('SNR')==20)&(pl.col("Ideal Sparsity")==0)&(pl.col("Lambda")==0.0)&(pl.col("SNR Type")==args.type)
        
    # palette =  sns.color_palette()[:2] + ["#8C8C8C"] + sns.color_palette()[2:4]
    
    latent_dim = df.filter(pl.col("Case").str.contains("Baseline"))["Symbols"].max()

    df_plot = df.filter(filter).with_columns(
                ((pl.col("Transmitting Antennas")/latent_dim)*100).alias("Semantic Compression Factor %")
            )
    
    ticks = df_plot.filter((~pl.col("Case").str.contains("Baseline", literal=True)))['Semantic Compression Factor %'].unique().round(2)
    
    # "Accuracy Vs Antennas" 
    plt.figure(figsize=(12, 8))
    plot = sns.lineplot(df_plot.to_pandas(), 
                        x='Semantic Compression Factor %', y='Accuracy', hue='Case', style="Case", dashes=dashes, markers=True, markersize=markersize, linewidth=linewidth).set(xlim=(ticks[0], ticks[-1]), ylim=(0, 1))#.set(xlim=(1, 12), ylim=(0, 1))# 
    plt.ylabel("Accuracy")
    plt.legend(loc=(0.18, 0.2))
    plt.savefig(str(IMG_PATH / 'accuracy_absolute.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    
    # ====================================================================================================================
    #                                        Accuracy Vs Signal to Noise Ratio
    # ====================================================================================================================
    filter = (pl.col('Transmitting Antennas')==8)&(pl.col('Receiving Antennas')==8)&(pl.col("Awareness")=="aware")&(pl.col('SNR')!=0)&(pl.col("Sparsity")==0.0)&(pl.col("SNR Type")==args.type)
    snr_df = df.filter(filter)

    # palette =  sns.color_palette()[:2] + ["#8C8C8C"]
        
    plt.figure(figsize=(12, 8))
    plot = sns.lineplot(snr_df.to_pandas(), 
                        x='SNR', y='Accuracy', hue='Case', style="Case",  markers=True, dashes=dashes, markersize=markersize, linewidth=linewidth).set(xlim=(-20, 30), ylim=(0, 1))
    plt.xlabel("Signal to Noise Ratio (dB)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(str(IMG_PATH / 'snr_zoom_absolute.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    
    # ====================================================================================================================
    #                                        Accuracy Vs Sparsity
    # ====================================================================================================================
    filter = (
        (pl.col("Awareness")=="aware") &
        (pl.col('SNR')==20) &
        (~pl.col("Case").str.contains("Baseline")) &
        (pl.col("Ideal Sparsity")==0) &
        (pl.col("SNR Type")==args.type)
    )
    
    tmp_df = (
        df
        .filter(filter &  (pl.col('Transmitting Antennas')==6) & (pl.col('Receiving Antennas')==6))
        .group_by(["Case", "Lambda"])
        .agg(pl.col("FLOPs").mean(), pl.col("Accuracy").mean())
        .select(["Case", "FLOPs", "Accuracy"])
        .with_columns(
                      pl.when(pl.col("Case")=="Neural Semantic Precoding/Decoding")
                      .then(pl.lit(r"Neural Semantic PDG with Hard Thresholding $\zeta=3\%$"))
                      .otherwise(pl.col("Case"))
                      .alias("Case"))
        .with_columns(
                      pl.when(pl.col("Case")=="Linear Semantic Precoding/Decoding")
                      .then(pl.lit(r"Linear Semantic $\zeta=3\%$"))
                      .otherwise(pl.col("Case"))
                      .alias("Case")
                  )
    )
    sparsity_df = (
        df
        .filter(filter &  (pl.col('Transmitting Antennas')==10) & (pl.col('Receiving Antennas')==10))
        .group_by(["Case", "Lambda"])
        .agg(pl.col("FLOPs").mean(), pl.col("Accuracy").mean())
        .select(["Case", "FLOPs", "Accuracy"])
        .with_columns(
                      pl.when(pl.col("Case")=="Neural Semantic Precoding/Decoding")
                      .then(pl.lit(r"Neural Semantic PDG with Hard Thresholding $\zeta=5\%$"))
                      .otherwise(pl.col("Case"))
                      .alias("Case"))
        .with_columns(
                      pl.when(pl.col("Case")=="Linear Semantic Precoding/Decoding")
                      .then(pl.lit(r"Linear Semantic $\zeta=5\%$"))
                      .otherwise(pl.col("Case"))
                      .alias("Case")
                  )
    )

    sparsity_df = (
        tmp_df
        .vstack(sparsity_df)
        # .with_columns(pl.col("FLOPs")/pl.col("FLOPs").max())
        # .with_columns(pl.col("FLOPs")*100)
    )
    
    plt.figure(figsize=(12, 8))
    plot = sns.lineplot(sparsity_df.to_pandas(), 
                        x='FLOPs', y='Accuracy', hue='Case', style="Case",  markers=True, markersize=markersize, linewidth=linewidth).set(ylim=(0, 1))
    plt.xlabel("FLOPS")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(str(IMG_PATH / 'accuracy_vs_flops.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    
    return None

if __name__ == "__main__":
    main()
