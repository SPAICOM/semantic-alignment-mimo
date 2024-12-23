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
                        default="final_results.parquet")

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

    # Get the baseline value
    baseline = df.filter(
                    (pl.col("Case").str.contains("Baseline")) &
                    (pl.col("Awareness")=="aware") &
                    (pl.col("SNR")==20)
                )["Accuracy"].mean()
            
    # Set the font to Times-Roman (or Times New Roman)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times']

    plt.rcParams.update({
                        'font.size': 12,  # General font size
                        'axes.titlesize': 14,  # Title font size
                        'axes.labelsize': 14,  # X and Y label size
                        'xtick.labelsize': 12,  # X-tick label size
                        'ytick.labelsize': 12,  # Y-tick label size
                        'legend.fontsize': 10,  # Legend font size
                        'legend.title_fontsize': 12  # Legend title font size
                        })
    
    # ====================================================================================================================
    #                                        Antennas Absolute
    # ====================================================================================================================
    filter = (pl.col('SNR')==20)
        
    palette =  sns.color_palette()[:2] + ["#8C8C8C"] + sns.color_palette()[2:4]
    
    latent_dim = df.filter(pl.col("Case").str.contains("Baseline"))["Symbols"].unique().item()

    df_plot = df.filter(filter).with_columns(
                pl.when(pl.col("Case").str.contains("Baseline"))
                .then(baseline)
                .otherwise(pl.col("Accuracy"))
                .alias("Accuracy"),
                ((pl.col("Transmitting Antennas")/latent_dim)*100).alias("Semantic Compression Factor %")
            )
    
    ticks = df_plot.filter((~pl.col("Case").str.contains("Baseline", literal=True)))['Semantic Compression Factor %'].unique().round(2)
    
    # "Accuracy Vs Antennas" 
    plot = sns.lineplot(df_plot.to_pandas(), 
                        x='Semantic Compression Factor %', y='Accuracy', hue='Case', style="Case", dashes=dashes, markers=True, markersize=10, palette=palette).set(xlim=(ticks[0], ticks[-1]), ylim=(0, 1)) 
    plt.ylabel("Accuracy")
    plt.legend(loc=(0.25, 0.25))
    plt.savefig(str(IMG_PATH / 'accuracy_absolute.pdf'), format='pdf')
    plt.show()
    
    # ====================================================================================================================
    #                                        Accuracy Vs Signal to Noise Ratio
    # ====================================================================================================================
    filter = (pl.col('Transmitting Antennas')==8)&(pl.col('Receiving Antennas')==8)&(pl.col("Awareness")=="aware")&(pl.col('SNR')!=0)
    snr_df = df.filter(filter)

    palette =  sns.color_palette()[:2] + ["#8C8C8C"]
        
    plot = sns.lineplot(snr_df.to_pandas(), 
                        x='SNR', y='Accuracy', hue='Case', style="Case",  markers=True, dashes=dashes, markersize=10, palette=palette).set(xlim=(-20, 30), ylim=(0, 1), xlabel="Signal to Noise Ratio (dB)")
    plt.xlabel("Signal to Noise Ratio (dB)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(str(IMG_PATH / 'snr_zoom_absolute.pdf'), format='pdf')
    plt.show()
    
    return None

if __name__ == "__main__":
    main()
