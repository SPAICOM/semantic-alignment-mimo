{
 description = "A Python 3.11 environment with PyTorch for scripts";

 inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

 outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      system = system;
    };
  in {
    devShell.${system} = with pkgs; mkShell {
      buildInputs = [
        (python311.withPackages (ps: with ps; [
          gdown
          numpy
          pip
          polars
          pyarrow
          python-dotenv
          pytorch-lightning
          scipy
          seaborn
          timm
          torch
          wandb
        ]))
      ];
    };
  };
}
