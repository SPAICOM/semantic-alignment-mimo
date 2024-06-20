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
    PARQUET_PATH: Path = CURRENT / 'final_results_with_noise.parquet'
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
    ticks = df['Transmitting Antennas'].unique()
    plot = sns.lineplot(df.filter((pl.col('Case').str.contains('abs'))&(pl.col('Receiving Antennas')==200)).to_pandas(), 
                        x='Transmitting Antennas', y='Accuracy', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title="Accuracy Vs Transmitting Antennas", ylim=(0, 1))
    plt.savefig(str(IMG_PATH / 'accuracy_transmitting_absolute.png'))
    plt.show()
    
    plot = sns.lineplot(df.filter((pl.col('Case').str.contains('abs'))&(pl.col('Receiving Antennas')==200)).to_pandas(), 
                        x='Transmitting Antennas', y='Classifier Loss', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title="Classifier Loss Vs Transmitting Antennas")
    plt.savefig(str(IMG_PATH / 'classifier_transmitting_absolute.png'))
    plt.show()
    
    plot = sns.lineplot(df.filter((pl.col('Case').str.contains('abs'))&(pl.col('Receiving Antennas')==200)).to_pandas(), 
                        x='Transmitting Antennas', y='Alignment Loss', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title="Alignment Loss Vs Transmitting Antennas")
    plt.savefig(str(IMG_PATH / 'alignment_transmitting_absolute.png'))
    plt.show()
    
    # ====================================================================================================================
    #                                        Receiving Antennas Absolute
    # ====================================================================================================================
    ticks = df['Receiving Antennas'].unique()
    plot = sns.lineplot(df.filter((pl.col('Case').str.contains('abs'))&(pl.col('Transmitting Antennas')==200)).to_pandas(), 
                        x='Receiving Antennas', y='Accuracy', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title="Accuracy Vs Receiving Antennas", ylim=(0, 1))
    plt.savefig(str(IMG_PATH / 'accuracy_receving_absolute.png'))
    plt.show()
    
    plot = sns.lineplot(df.filter((pl.col('Case').str.contains('abs'))&(pl.col('Transmitting Antennas')==200)).to_pandas(), 
                        x='Receiving Antennas', y='Classifier Loss', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title="Classifier Loss Vs Receiving Antennas")
    plt.savefig(str(IMG_PATH / 'classifier_receiving_absolute.png'))
    plt.show()
    
    plot = sns.lineplot(df.filter((pl.col('Case').str.contains('abs'))&(pl.col('Transmitting Antennas')==200)).to_pandas(), 
                        x='Receiving Antennas', y='Alignment Loss', hue='Case').set(xlim=(ticks[0], ticks[-1]), xticks=ticks, title="Alignment Loss Vs Receiving Antennas")
    plt.savefig(str(IMG_PATH / 'alignment_receiving_absolute.png'))
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
