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
    df = pl.read_parquet(PARQUET_PATH)

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
                        'legend.title_fontsize': 14  # Legend title font size
                        })
    
    # ====================================================================================================================
    #                                        Antennas Absolute
    # ====================================================================================================================
    filter = (pl.col('Sigma')==0.1)
    ticks = df.filter(filter)['Transmitting Antennas'].unique()
    
    # "Accuracy Vs Antennas" 
    plot = sns.lineplot(df.filter(filter).to_pandas(), 
                        x='Transmitting Antennas', y='Accuracy', hue='Case', style="Case",  markers=True, dashes=False, markersize=10).set(xlim=(ticks[0], ticks[-1]), xticks=ticks, ylim=(0, 1), xlabel="Number of Antennas")
    plt.xlabel("Number of Antennas", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc=(0.25, 0.28))
    plt.savefig(str(IMG_PATH / 'accuracy_absolute.pdf'), format='pdf')
    plt.show()
    
    # "Accuracy Vs Antennas Normalized"
    plot = sns.lineplot(df.filter(filter).with_columns(pl.col("Accuracy")*(pl.col("Transmitting Antennas")/pl.col("Simbols"))).to_pandas(), 
                        x='Transmitting Antennas', y='Accuracy', hue='Case', style="Case",  markers=True, dashes=False, markersize=10).set(xlim=(ticks[0], ticks[-1]), xticks=ticks, ylim=(0, 1), xlabel="Number of Antennas")
    plt.xlabel("Number of Antennas", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc=(0.25, 0.28))
    plt.savefig(str(IMG_PATH / 'accuracy_absolute_normalized.pdf'), format='pdf')
    plt.show()
    
    plot = sns.lineplot(df.filter(filter).to_pandas(), 
                        x='Transmitting Antennas', y='Classifier Loss', hue='Case', style="Case",  markers=True, dashes=False, markersize=10).set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title=f"Classifier Loss Vs Antennas", xlabel="Number of Antennas")
    plt.savefig(str(IMG_PATH / 'classifier_absolute.png'))
    plt.show()
    
    plot = sns.lineplot(df.filter(filter).to_pandas(), 
                        x='Transmitting Antennas', y='Alignment Loss', hue='Case', style="Case",  markers=True, dashes=False, markersize=10).set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title=f"Alignment Loss Vs Antennas in log scale", yscale='log', xlabel="Number of Antennas")
    plt.savefig(str(IMG_PATH / 'alignment_log_absolute.png'))
    plt.show()
    
    filter = filter & (pl.col('Case').str.contains(' Aware'))
    plot = sns.lineplot(df.filter(filter).to_pandas(), 
                        x='Transmitting Antennas', y='Alignment Loss', hue='Case', style="Case",  markers=True, dashes=False, markersize=10).set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title=f"Alignment Loss Vs Antennas Only Aware", xlabel="Number of Antennas")
    plt.savefig(str(IMG_PATH / 'alignment_absolute.png'))
    plt.show()
    
    
    # ====================================================================================================================
    #                                        Accuracy Vs Signal to Noise Ratio
    # ====================================================================================================================
    filter = (pl.col('Transmitting Antennas')==8)&(pl.col('Receiving Antennas')==8)&(pl.col('Case').str.contains(' Aware'))
    snr_df = df.filter(filter).group_by(["Case", "Sigma"], maintain_order=True).agg(pl.col("SNR").mean(), pl.col("Accuracy"), pl.col("Transmitting Antennas"), pl.col("Simbols")).explode(["Accuracy", "Transmitting Antennas", "Simbols"])
    
    plot = sns.lineplot(snr_df.to_pandas(), 
                        x='SNR', y='Accuracy', hue='Case', style="Case",  markers=True, dashes=False, markersize=10).set(ylim=(0, 1), title="Accuracy Vs Signal to Noise Ratio (Tx=8, Rx=8)", xlabel="Signal to Noise Ratio (dB)")
    plt.savefig(str(IMG_PATH / 'snr_absolute.png'))
    plt.show()
    
    plot = sns.lineplot(snr_df.to_pandas(), 
                        x='SNR', y='Accuracy', hue='Case', style="Case",  markers=True, dashes=False, markersize=10).set(xlim=(-20, 30), ylim=(0, 1), xlabel="Signal to Noise Ratio (dB)")
    plt.xlabel("Signal to Noise Ratio (dB)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig(str(IMG_PATH / 'snr_zoom_absolute.pdf'), format='pdf')
    plt.show()
    
    plot = sns.lineplot(snr_df.with_columns(pl.col("Accuracy")*(pl.col("Transmitting Antennas")/pl.col("Simbols"))).to_pandas(), 
                        x='SNR', y='Accuracy', hue='Case', style="Case",  markers=True, dashes=False, markersize=10).set(xlim=(-20, 30), ylim=(0, 1), xlabel="Signal to Noise Ratio (dB)")
    plt.xlabel("Signal to Noise Ratio (dB)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig(str(IMG_PATH / 'snr_zoom_absolute_normalized.pdf'), format='pdf')
    plt.show()
    
    ticks = df.filter(filter)['Sigma'].unique()
       
    plot = sns.lineplot(snr_df.to_pandas(), 
                        x='Sigma', y='Accuracy', hue='Case', style="Case",  markers=True, dashes=False, markersize=10).set(xscale='log', ylim=(0, 1), xlim=(ticks[0], ticks[-1]), title="Accuracy Vs Sigma (Tx=8, Rx=8)")
    plt.savefig(str(IMG_PATH / 'sigma_absolute.png'))
    plt.show()
    
    return None

if __name__ == "__main__":
    main()
