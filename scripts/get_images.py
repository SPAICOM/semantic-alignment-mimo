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

    # Set style
    plt.style.use('.conf/plotting/plt.mplstyle')
    
    # Read the parquet
    df = (
        pl.read_parquet(PARQUET_PATH).sort(by=["Awareness", "Case"], descending=[False, True])
        .with_columns(
                      pl.when(pl.col("Case").str.contains("Baseline"))
                      .then((5, 5))
                      .otherwise(())
                      .alias("dashes")
        )
        .filter(pl.col("Symbols")<=pl.col("Transmitting Antennas").max())
    )

    dashes = df.select(["Case", "dashes"]).unique(subset=["Case"]).to_dict(as_series= False)
    dashes = dict(zip(dashes["Case"], dashes["dashes"]))

    # ====================================================================================================================
    #                                        Antennas Absolute
    # ====================================================================================================================
    filter = (pl.col('SNR')==20)&(pl.col("Ideal Sparsity")==0)&(pl.col("Lambda")==0.0)&(pl.col("SNR Type")==args.type)&(pl.col("Symbols")!=0)
        
    # palette =  sns.color_palette()[:2] + ["#8C8C8C"] + sns.color_palette()[2:4]
    
    latent_dim = df["Symbols"].max()

    df_plot = df.filter(filter).with_columns(
                ((pl.col("Symbols")/latent_dim)*100).alias("Compression Factor")
            )
    
    ticks = df_plot.filter((~pl.col("Case").str.contains("Baseline", literal=True)))['Compression Factor'].unique().sort(descending=False).round(2)
    fake_ticks = [5e-1, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]
    
    # "Accuracy Vs Antennas" 
    plt.figure(figsize=(16, 10))
    plot = sns.lineplot(df_plot.to_pandas(), 
                        x='Compression Factor', y='Accuracy', hue='Case', style="Case", dashes=dashes, markers=True).set(xticks=ticks, xlim=(5e-1, ticks[-1]), ylim=(0, 1), xscale="log")
    plt.ylabel("Accuracy")
    plt.xlabel(r'Compression Factor $\zeta$ (\%)')
    plt.xticks(fake_ticks, labels=fake_ticks)
    plt.legend(loc="center", bbox_to_anchor=(0.5, 1.1), ncol=2)  # Adjust bbox_to_anchor as needed
    plt.savefig(str(IMG_PATH / 'accuracy_absolute.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    
    # "Zoom on Accuracy Vs Antennas"
    plt.figure(figsize=(16, 10))
    plot = sns.lineplot(df_plot.to_pandas(), 
                        x='Compression Factor', y='Accuracy', hue='Case', style="Case", dashes=dashes, markers=True).set(xlim=(ticks[0], 8), ylim=(0, 1))
    plt.ylabel("Accuracy")
    plt.xlabel(r'Compression Factor $\zeta$ (\%)')
    plt.legend(loc="center", bbox_to_anchor=(0.5, 1.1), ncol=2)  # Adjust bbox_to_anchor as needed
    plt.savefig(str(IMG_PATH / 'accuracy_absolute_zoom.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    
    # ====================================================================================================================
    #                                        Accuracy Vs Signal to Noise Ratio
    # ====================================================================================================================
    filter = (pl.col('Transmitting Antennas')==8)&(pl.col('Receiving Antennas')==8)&(pl.col("Awareness")=="aware")&(pl.col('SNR')!=0)&(pl.col("Sparsity")==0.0)&(pl.col("SNR Type")==args.type)
    snr_df = df.filter(filter)

    # palette =  sns.color_palette()[:2] + ["#8C8C8C"]
        
    plt.figure(figsize=(12, 8))
    plot = sns.lineplot(snr_df.to_pandas(), 
                        x='SNR', y='Accuracy', hue='Case', style="Case",  markers=True, dashes=dashes).set(xlim=(-20, 30), ylim=(0, 1))
    plt.xlabel("Signal to Noise Ratio (dB)")
    plt.ylabel("Accuracy")
    plt.legend(loc="center", bbox_to_anchor=(0.5, 1.1), ncol=2)  # Adjust bbox_to_anchor as needed
    plt.savefig(str(IMG_PATH / 'snr_zoom_absolute.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    
    # ====================================================================================================================
    #                                        Accuracy Vs FLOPS
    # ====================================================================================================================
    filter = (
        (pl.col("Awareness")=="aware") &
        (pl.col('SNR')==20) &
        (~pl.col("Case").str.contains("Baseline")) &
        (pl.col("Ideal Sparsity")==0) &
        (pl.col("SNR Type")==args.type)
    )
    
    df_6x6 = (
        df
        .filter(filter &  (pl.col('Transmitting Antennas')==6) & (pl.col('Receiving Antennas')==6))
        .group_by(["Case", "Lambda"])
        .agg(pl.col("FLOPs").mean(), pl.col("Accuracy").mean())
        .select(["Case", "FLOPs", "Accuracy"])
        .with_columns(
                      pl.when(pl.col("Case")=="Neural Semantic Precoding/Decoding")
                      .then(pl.lit(r"Neural Semantic $\zeta=3\%$ PDG with Hard Thresholding"))
                      .otherwise(pl.col("Case"))
                      .alias("Case"))
        .with_columns(
                      pl.when(pl.col("Case")=="Linear Semantic Precoding/Decoding")
                      .then(pl.lit(r"Linear Semantic $\zeta=3\%$"))
                      .otherwise(pl.col("Case"))
                      .alias("Case")
                  )
    ).sort(["Case"], descending=True)
    df_10x10 = (
        df
        .filter(filter &  (pl.col('Transmitting Antennas')==10) & (pl.col('Receiving Antennas')==10))
        .group_by(["Case", "Lambda"])
        .agg(pl.col("FLOPs").mean(), pl.col("Accuracy").mean())
        .select(["Case", "FLOPs", "Accuracy"])
        .with_columns(
                      pl.when(pl.col("Case")=="Neural Semantic Precoding/Decoding")
                      .then(pl.lit(r"Neural Semantic $\zeta=5\%$ PDG with Hard Thresholding"))
                      .otherwise(pl.col("Case"))
                      .alias("Case"))
        .with_columns(
                      pl.when(pl.col("Case")=="Linear Semantic Precoding/Decoding")
                      .then(pl.lit(r"Linear Semantic $\zeta=5\%$"))
                      .otherwise(pl.col("Case"))
                      .alias("Case")
                  )
    ).sort(["Case"], descending=True)

    plot_df = (
        df_6x6
        .vstack(df_10x10)
        # .with_columns(pl.col("FLOPs")/pl.col("FLOPs").max())
        # .with_columns(pl.col("FLOPs")*100)
    )
    
    plt.figure(figsize=(12, 8))
    plot = sns.lineplot(plot_df.to_pandas(), 
                        x='FLOPs', y='Accuracy', hue='Case', style="Case", dashes=False, markers=True).set(ylim=(0, 1)) 
    plt.xlabel("FLOPs")
    plt.ylabel("Accuracy")
    # plt.legend(loc="center", bbox_to_anchor=(0.5, 1.15), ncol=2)  # Adjust bbox_to_anchor as needed
    plt.legend()
    plt.savefig(str(IMG_PATH / 'accuracy_vs_flops.pdf'), format='pdf', bbox_inches='tight')
    plt.show()


    
    # ====================================================================================================================
    #                                  Different channel usage and channel size
    # ====================================================================================================================
    filter = (pl.col('SNR')==20)&(pl.col("Ideal Sparsity")==0)&(pl.col("Lambda")==0.0)&(pl.col("SNR Type")==args.type)&(pl.col("Symbols")!=0)&(pl.col("Awareness")=="aware")
        
    # palette =  sns.color_palette()[:2] + ["#8C8C8C"] + sns.color_palette()[2:4]
    
    latent_dim = df["Symbols"].max()

    df_plot = (
        df.filter(
            filter &
            (pl.col("Case").str.starts_with("Linear "))
        )
        .with_columns(
            ((pl.col("Symbols")/latent_dim)*100).alias("Compression Factor"),
            pl.lit(r"$N_t = N_r = 1$").alias("Case")
        )
        .select(["Accuracy", "Compression Factor", "Case"])
    )
    
    compr_factor = set(df_plot["Compression Factor"].unique().to_list())
    
    final_plot = (
        df_plot.vstack(
            df_plot
            .with_columns(
                pl.col("Compression Factor")/2,
                pl.lit(r"$N_t = N_r = 2$").alias("Case")
            )
            .filter(
                pl.col("Compression Factor").is_in(compr_factor)
            )
        )
    )
    final_plot = (
        final_plot.vstack(
            df_plot
            .with_columns(
                pl.col("Compression Factor")/4,
                pl.lit(r"$N_t = N_r = 4$").alias("Case")
            )
            .filter(
                pl.col("Compression Factor").is_in(compr_factor)
            )
        )
    )
    final_plot = (
        final_plot.vstack(
            df_plot
            .with_columns(
                pl.col("Compression Factor")/8,
                pl.lit(r"$N_t = N_r = 8$").alias("Case")
            )
            .filter(
                pl.col("Compression Factor").is_in(compr_factor)
            )
        )
    )
    
    fake_ticks = [5e-1, 1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]
    
    # "Accuracy Vs Antennas" 
    plt.figure(figsize=(16, 10))
    plot = sns.lineplot(final_plot.to_pandas(), 
                        x='Compression Factor', y='Accuracy', hue="Case", markers=True).set(xticks=ticks, xlim=(5e-1, ticks[-1]), ylim=(0, 1), xscale="log")
    plt.ylabel("Accuracy")
    plt.xlabel(r'Compression Factor $\zeta$ (\%)')
    plt.xticks(fake_ticks, labels=fake_ticks)
    plt.legend(loc="center", bbox_to_anchor=(0.5, 1.1), ncol=2)  # Adjust bbox_to_anchor as needed
    plt.savefig(str(IMG_PATH / 'accuracy_K_N.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    
    return None

if __name__ == "__main__":
    main()
