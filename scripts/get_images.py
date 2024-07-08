"""A python module to create the visualizations.
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
    # Define some paths
    CURRENT: Path = Path('.')
    PARQUET_PATH: Path = CURRENT / 'final_results.parquet'
    IMG_PATH: Path = CURRENT / 'img'

    # Create the 'img' folder if it doesn't exist
    IMG_PATH.mkdir(parents=True, exist_ok=True)

    # Set the plot style in seaborn
    sns.set_style('whitegrid')
    
    # Read the parquet
    df = pl.read_parquet(PARQUET_PATH)

    # ====================================================================================================================
    #                                        Transmitting Antennas Absolute
    # ====================================================================================================================
    ticks = df.filter(pl.col('Transmitting Antennas')!=100)['Transmitting Antennas'].unique()

    filter = (pl.col('Case').str.contains('abs'))&(pl.col('Transmitting Antennas')!=100)
    
    plot = sns.lineplot(df.filter(filter).to_pandas(), 
                        x='Transmitting Antennas', y='Accuracy', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title=f"Accuracy Vs Antennas", ylim=(0, 1))
    plt.savefig(str(IMG_PATH / 'accuracy_absolute.png'))
    plt.show()
    
    plot = sns.lineplot(df.filter(filter).to_pandas(), 
                        x='Transmitting Antennas', y='Classifier Loss', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title=f"Classifier Loss Vs Antennas")
    plt.savefig(str(IMG_PATH / 'classifier_absolute.png'))
    plt.show()
    
    plot = sns.lineplot(df.filter(filter).to_pandas(), 
                        x='Transmitting Antennas', y='Alignment Loss', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title=f"Alignment Loss Vs Antennas in log scale", yscale='log')
    plt.savefig(str(IMG_PATH / 'alignment_log_absolute.png'))
    plt.show()
    
    filter = filter & (pl.col('Case').str.contains(' aware '))
    plot = sns.lineplot(df.filter(filter).to_pandas(), 
                        x='Transmitting Antennas', y='Alignment Loss', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title=f"Alignment Loss Vs Antennas Only Aware")
    plt.savefig(str(IMG_PATH / 'alignment_absolute.png'))
    plt.show()
    
    # ====================================================================================================================
    #                                        Receiving Antennas Absolute
    # ====================================================================================================================
    # ticks = df.filter(pl.col('Receiving Antennas')!=rc_fixed)['Receiving Antennas'].unique()
    
    # plot = sns.lineplot(df.filter((pl.col('Case').str.contains('abs'))&(pl.col('Transmitting Antennas')==tr_fixed)).to_pandas(), 
    #                     x='Receiving Antennas', y='Accuracy', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title=f"Accuracy Vs Receiving Antennas (Tx = {tr_fixed})", ylim=(0, 1))
    # plt.savefig(str(IMG_PATH / 'accuracy_receving_absolute.png'))
    # plt.show()
    
    # plot = sns.lineplot(df.filter((pl.col('Case').str.contains('abs'))&(pl.col('Transmitting Antennas')==tr_fixed)).to_pandas(), 
    #                     x='Receiving Antennas', y='Classifier Loss', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title=f"Classifier Loss Vs Receiving Antennas (Tx = {tr_fixed})")
    # plt.savefig(str(IMG_PATH / 'classifier_receiving_absolute.png'))
    # plt.show()
    
    # plot = sns.lineplot(df.filter((pl.col('Case').str.contains('abs'))&(pl.col('Transmitting Antennas')==tr_fixed)).to_pandas(), 
    #                     x='Receiving Antennas', y='Alignment Loss', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title=f"Alignment Loss Vs Receiving Antennas (Tx = {tr_fixed})")
    # plt.savefig(str(IMG_PATH / 'alignment_receiving_absolute.png'))
    # plt.show()

    
    # ====================================================================================================================
    #                                        Accuracy Vs Signal to Noise Ratio
    # ====================================================================================================================
    plot = sns.lineplot(df.filter((pl.col('Case').str.contains(' aware abs'))&(pl.col('Transmitting Antennas')==100)&(pl.col('Receiving Antennas')==100)&(pl.col("Sigma")!=0)).to_pandas(), 
                        x='SNR', y='Accuracy', hue='Case').set(ylim=(0, 1), title="Accuracy Vs Signal to Noise Ratio (Tx=100, Rx=100)", xlabel="Signal to Noise Ratio (dB)")
    plt.savefig(str(IMG_PATH / 'snr_absolute.png'))
    plt.show()
    
    plot = sns.lineplot(df.filter((pl.col('Case').str.contains(' aware abs'))&(pl.col('Transmitting Antennas')==100)&(pl.col('Receiving Antennas')==100)&(pl.col("Sigma")!=0)).to_pandas(), 
                        x='SNR', y='Accuracy', hue='Case').set(xlim=(-30, 20), ylim=(0, 1), title="Accuracy Vs Signal to Noise Ratio (Tx=100, Rx=100)", xlabel="Signal to Noise Ratio (dB)")
    plt.savefig(str(IMG_PATH / 'snr_zoom_absolute.png'))
    plt.show()
    
    ticks = df.filter((pl.col('Case').str.contains('abs'))&(pl.col('Transmitting Antennas')==100)&(pl.col('Receiving Antennas')==100))['Sigma'].unique()
       
    plot = sns.lineplot(df.filter((pl.col('Case').str.contains(' aware abs'))&(pl.col('Transmitting Antennas')==100)&(pl.col('Receiving Antennas')==100)&(pl.col('Sigma')!=0)).to_pandas(), 
                        x='Sigma', y='Accuracy', hue='Case').set(xscale='log', ylim=(0, 1), xlim=(ticks[0], ticks[-1]), title="Accuracy Vs Sigma (Tx=100, Rx=100)")
    plt.savefig(str(IMG_PATH / 'sigma_absolute.png'))
    plt.show()
    
    # # ====================================================================================================================
    # #                                        Transmitting Antennas Relative
    # # ====================================================================================================================
    # ticks = df['Transmitting Antennas'].unique()
    # plot = sns.FacetGrid(df.filter((pl.col('Case').str.contains('rel'))&(pl.col('Receiving Antennas')==200)).to_pandas(), hue='Case', col="Anchors")
    # plot.map(sns.lineplot, 'Transmitting Antennas', 'Accuracy')
    # plot.set(xlim=(ticks[0], ticks[-1]), xticks=ticks, ylim=(0, 1))
    # plot.add_legend()
    # plot.figure.subplots_adjust(wspace=0.1)
    # plt.savefig(str(IMG_PATH / 'accuracy_transmitting_relative.png'))
    # plt.show()
    
    # plot = sns.FacetGrid(df.filter((pl.col('Case').str.contains('rel'))&(pl.col('Receiving Antennas')==200)).to_pandas(), hue='Case', col="Anchors")
    # plot.map(sns.lineplot, 'Transmitting Antennas', 'Classifier Loss')
    # plot.set(xlim=(ticks[0], ticks[-1]), xticks=ticks)
    # plot.add_legend()
    # plot.figure.subplots_adjust(wspace=0.1)
    # plt.savefig(str(IMG_PATH / 'classifier_transmitting_relative.png'))
    # plt.show()
    
    # plot = sns.FacetGrid(df.filter((pl.col('Case').str.contains('rel'))&(pl.col('Receiving Antennas')==200)).to_pandas(), hue='Case', col="Anchors")
    # plot.map(sns.lineplot, 'Transmitting Antennas', 'Alignment Loss')
    # plot.set(xlim=(ticks[0], ticks[-1]), xticks=ticks)
    # plot.add_legend()
    # plot.figure.subplots_adjust(wspace=0.1)
    # plt.savefig(str(IMG_PATH / 'alignment_transmitting_relative.png'))
    # plt.show()
    
    # # ====================================================================================================================
    # #                                        Receiving Antennas Relative
    # # ====================================================================================================================
    # ticks = df['Receiving Antennas'].unique()
    # plot = sns.FacetGrid(df.filter((pl.col('Case').str.contains('rel'))&(pl.col('Transmitting Antennas')==200)).to_pandas(), hue='Case', col="Anchors")
    # plot.map(sns.lineplot, 'Receiving Antennas', 'Accuracy')
    # plot.set(xlim=(ticks[0], ticks[-1]), xticks=ticks, ylim=(0, 1))
    # plot.add_legend()
    # plot.figure.subplots_adjust(wspace=0.1)
    # plt.savefig(str(IMG_PATH / 'accuracy_receving_relative.png'))
    # plt.show()
    
    # plot = sns.FacetGrid(df.filter((pl.col('Case').str.contains('rel'))&(pl.col('Transmitting Antennas')==200)).to_pandas(), hue='Case', col="Anchors")
    # plot.map(sns.lineplot, 'Receiving Antennas', 'Classifier Loss')
    # plot.set(xlim=(ticks[0], ticks[-1]), xticks=ticks)
    # plot.add_legend()
    # plot.figure.subplots_adjust(wspace=0.1)
    # plt.savefig(str(IMG_PATH / 'classifier_receving_relative.png'))
    # plt.show()
    
    # plot = sns.FacetGrid(df.filter((pl.col('Case').str.contains('rel'))&(pl.col('Transmitting Antennas')==200)).to_pandas(), hue='Case', col="Anchors")
    # plot.map(sns.lineplot, 'Receiving Antennas', 'Alignment Loss')
    # plot.set(xlim=(ticks[0], ticks[-1]), xticks=ticks)
    # plot.add_legend()
    # plot.figure.subplots_adjust(wspace=0.1)
    # plt.savefig(str(IMG_PATH / 'alignment_receving_relative.png'))
    # plt.show()
        
    return None

if __name__ == "__main__":
    main()
