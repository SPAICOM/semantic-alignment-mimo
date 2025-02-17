"""This python module handles the download of the trained models from wandb.

    To check available parameters run 'python /path/to/wandb_model_downloader.py --help'.
"""
    
import wandb
import shutil
from pathlib import Path
from tqdm.auto import tqdm


def find_and_rename_ckpt_files(root_dir: str,
                               new_dir: str,
                               name: str) -> None:
    """Save the results in a convenient directory structure.

    Args:
        root_dir : str
            The directory where the files are currently saved.
        new_dir : str
            The new directory where to save the files.
        name : str
            The new name of the file.

    Returns:
        None
    """
    root_dir = Path(root_dir)
    new_dir = Path(new_dir)

    # Ensure the new directory exists
    new_dir.mkdir(parents=True, exist_ok=True)

    # Find all .ckpt files in subdirectories
    ckpt_files = list(root_dir.rglob('*.ckpt'))

    new_path = new_dir / f'seed_{name}.ckpt'
    shutil.move(str(ckpt_files[0]), str(new_path))
    print(f'Moved: {ckpt_files[0]} to {new_path}')
    
    return None


        
# ============================================================
#
#                     MAIN DEFINITION
#
# ============================================================

def main() -> None:
    """The main loop.
    """
    import argparse
    
    description = """
    This python module handles the download of the trained models from wandb.

    To check available parameters run 'python /path/to/wandb_model_downloader.py --help'.
    """
    
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-o',
                        '--org',
                        help="The organization of wandb where the models are saved.",
                        type=str)
    
    parser.add_argument('--pruned',
                        help="If to download the pruned version of the models or not. Default False.",
                        default=False,
                        type=bool,
                        action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()

    run = wandb.init()

    seeds = [27, 42, 100, 123, 144, 200]
    antennas = [1, 2, 4, 6, 8, 10, 12, 16, 24, 48, 96, 192] 
    snr = [-20.0, -10.0, 10.0, 30.0]
    snr_types = ["transmitted", "received"]
    sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    lmbs = [1e1, 2e1, 3e1, 4e1, 5e1, 6e1, 7e1, 8e1]
    costs = [1]
    
    if args.pruned:
        for cost in costs:
            for seed in seeds:
                for sparsity in sparsities:
                    run.use_artifact(f'{args.org}/SemanticAutoEncoder_wn_6_6_aware_20.0_{cost}_0.0_pruned_{sparsity}/model-seed_{seed}:best', type='model').download()
                    find_and_rename_ckpt_files('artifacts', f'aware-{cost}/antennas_6_6/snr_20.0/pruned_{sparsity}/', str(seed))

    else:
        # Just the lambdas
        for lmb in tqdm(lmbs):
            for cost in costs:
                for snr_type in snr_types:
                    for seed in seeds:
                        run.use_artifact(f'{args.org}/SemanticAutoEncoder_wn_6_6_aware_{snr_type}_20.0_{cost}_{lmb}/model-seed_{seed}:best', type='model').download()
                        find_and_rename_ckpt_files('artifacts', f'aware-{cost}/antennas_6_6/{snr_type}/snr_20.0/lmb_{lmb}/', str(seed))
                    
                        run.use_artifact(f'{args.org}/SemanticAutoEncoder_wn_10_10_aware_{snr_type}_20.0_{cost}_{lmb}/model-seed_{seed}:best', type='model').download()
                        find_and_rename_ckpt_files('artifacts', f'aware-{cost}/antennas_10_10/{snr_type}/snr_20.0/lmb_{lmb}/', str(seed))
                    
        # Not sparse solutions
        for ant in tqdm(antennas):
            for snr_type in snr_types:
                for cost in costs:
                    for seed in seeds:
                        run.use_artifact(f'{args.org}/SemanticAutoEncoder_wn_{ant}_{ant}_aware_{snr_type}_20.0_{cost}_0.0/model-seed_{seed}:best', type='model').download()
                        find_and_rename_ckpt_files('artifacts', f'aware-{cost}/antennas_{ant}_{ant}/{snr_type}/snr_20.0/lmb_0.0/', str(seed))

                    for seed in seeds:
                        run.use_artifact(f'{args.org}/SemanticAutoEncoder_wn_{ant}_{ant}_unaware_{snr_type}_None_{cost}_0.0/model-seed_{seed}:best', type='model').download()
                        find_and_rename_ckpt_files('artifacts', f'unaware-{cost}/antennas_{ant}_{ant}/{snr_type}/snr_20.0/lmb_0.0/', str(seed))

        for s in tqdm(snr):
            for snr_type in snr_types:
                for cost in costs:
                    for seed in seeds:
                        run.use_artifact(f'{args.org}/SemanticAutoEncoder_wn_8_8_aware_{snr_type}_{s}_{cost}_0.0/model-seed_{seed}:best', type='model').download()
                        find_and_rename_ckpt_files('artifacts', f'aware-{cost}/antennas_8_8/{snr_type}/snr_{s}/lmb_0.0/', str(seed))
            
    wandb.finish()
    
    return None



if __name__ == "__main__":
    main()
