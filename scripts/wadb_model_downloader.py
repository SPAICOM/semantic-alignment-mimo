import wandb
import shutil
from pathlib import Path
from tqdm.auto import tqdm


def find_and_rename_ckpt_files(root_dir, new_dir, name):
    root_dir = Path(root_dir)
    new_dir = Path(new_dir)

    # Ensure the new directory exists
    new_dir.mkdir(parents=True, exist_ok=True)

    # Find all .ckpt files in subdirectories
    ckpt_files = list(root_dir.rglob('*.ckpt'))

    # # Sort files by last modified time in descending order
    # ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # # Define the new names
    # new_names = [f'seed_{i}' for i in [42, 123, 200]]
    
    # # Move and rename files
    # for i, ckpt_file in enumerate(ckpt_files[:3]):  # Limit to first 3 files
    #     new_path = new_dir / f'{new_names[i]}.ckpt'
    #     shutil.move(str(ckpt_file), str(new_path))
    #     print(f'Moved: {ckpt_file} to {new_path}')
    new_path = new_dir / f'seed_{name}.ckpt'
    shutil.move(str(ckpt_files[0]), str(new_path))
    print(f'Moved: {ckpt_files[0]} to {new_path}')
    
    return None

        
def main() -> None:
    """
    """
    run = wandb.init()

    seeds = [42, 123, 200]
    antennas = [5, 10, 20, 40, 80, 120, 180, 250]
    sigmas = [0.01, 0.1, 1., 10., 100., 1000.]
    costs = ['None', 1000]
    costs = [1000]

    for ant in tqdm(antennas):
        for cost in costs:
            for seed in seeds:
                run.use_artifact(f'jrhin-org/SemanticAutoEncoder_wn_sparse_abs_abs_{ant}_{ant}_unaware_0_{cost}/model-seed_{seed}:best', type='model').download()

                # find_and_rename_ckpt_files('artifacts', f'unaware-{cost}/antennas_{ant}_{ant}/sigma_1.0/', seed)
                find_and_rename_ckpt_files('artifacts', f'unaware-sparse/antennas_{ant}_{ant}/sigma_1.0/', seed)

            for seed in seeds:
                run.use_artifact(f'jrhin-org/SemanticAutoEncoder_wn_sparse_abs_abs_{ant}_{ant}_aware_1.0_{cost}/model-seed_{seed}:best', type='model').download()

                # find_and_rename_ckpt_files('artifacts', f'aware-{cost}/antennas_{ant}_{ant}/sigma_1.0/', seed)
                find_and_rename_ckpt_files('artifacts', f'aware-sparse/antennas_{ant}_{ant}/sigma_1.0/', seed)

   
    # for ant in tqdm(antennas):
    #     break
    #     for seed in seeds:
    #         run.use_artifact(f'jrhin-org/SemanticAutoEncoder_wn_abs_abs_350_{ant}_unaware_0/cost-1000_seed-{seed}:best', type='model').download()
            
    #     find_and_rename_ckpt_files('artifacts', f'unaware-1000/antennas_350_{ant}/sigma_1.0/')
            
    #     for seed in seeds:
    #         run.use_artifact(f'jrhin-org/SemanticAutoEncoder_wn_abs_abs_350_{ant}_aware_1/cost-1000_seed-{seed}:best', type='model').download()
            
    #     find_and_rename_ckpt_files('artifacts', f'aware-1000/antennas_350_{ant}/sigma_1.0/')

    #     for seed in seeds:
    #         run.use_artifact(f'jrhin-org/SemanticAutoEncoder_wn_abs_abs_350_{ant}_unaware_0/cost-0_seed-{seed}:best', type='model').download()
            
    #     find_and_rename_ckpt_files('artifacts', f'unaware/antennas_350_{ant}/sigma_1.0/')
            
    #     for seed in seeds:
    #         run.use_artifact(f'jrhin-org/SemanticAutoEncoder_wn_abs_abs_350_{ant}_aware_1/cost-0_seed-{seed}:best', type='model').download()
            
    #     find_and_rename_ckpt_files('artifacts', f'aware/antennas_350_{ant}/sigma_1.0/')

    for sigma in tqdm(sigmas):
        for cost in costs:
            for seed in seeds:
                run.use_artifact(f'jrhin-org/SemanticAutoEncoder_wn_sparse_abs_abs_100_100_aware_{sigma}_{cost}/model-seed_{seed}:best', type='model').download()
            
                # find_and_rename_ckpt_files('artifacts', f'aware-{cost}/antennas_100_100/sigma_{sigma}/', seed)
                find_and_rename_ckpt_files('artifacts', f'aware-sparse/antennas_100_100/sigma_{sigma}/', seed)
            
    wandb.finish()
    
    return None

if __name__ == "__main__":
    main()
