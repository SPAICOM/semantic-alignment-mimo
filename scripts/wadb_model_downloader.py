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
    run = wandb.init()

    seeds = [27, 42, 100, 123, 144, 200]
    antennas = [1, 2, 4, 6, 8, 10, 12, 14] 
    sigmas = [0.0001, 0.001, 0.01, 0.1, 10., 100., 1000., 0.0005, 0.005, 0.05, 0.5, 5., 50., 500.]
    costs = [1]

    for ant in tqdm(antennas):
        for cost in costs:
            for seed in seeds:
                run.use_artifact(f'jrhin-org/SemanticAutoEncoder_wn_{ant}_{ant}_unaware_0_{cost}/model-seed_{seed}:best', type='model').download()

                find_and_rename_ckpt_files('artifacts', f'unaware-{cost}/antennas_{ant}_{ant}/sigma_0.1/', str(seed))

            for seed in seeds:
                run.use_artifact(f'jrhin-org/SemanticAutoEncoder_wn_{ant}_{ant}_aware_0.1_{cost}/model-seed_{seed}:best', type='model').download()

                find_and_rename_ckpt_files('artifacts', f'aware-{cost}/antennas_{ant}_{ant}/sigma_0.1/', str(seed))

    for sigma in tqdm(sigmas):
        for cost in costs:
            for seed in seeds:
                run.use_artifact(f'jrhin-org/SemanticAutoEncoder_wn_8_8_aware_{sigma}_{cost}/model-seed_{seed}:best', type='model').download()
            
                find_and_rename_ckpt_files('artifacts', f'aware-{cost}/antennas_8_8/sigma_{sigma}/', str(seed))
            
    wandb.finish()
    
    return None



if __name__ == "__main__":
    main()
