{
 description = "A Python 3.11 environment with PyTorch for scripts";

 inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

 outputs = { self, nixpkgs }: {
    devShell.x86_64-linux = with nixpkgs.legacyPackages.x86_64-linux; mkShell {
      buildInputs = [
        (python311.withPackages (ps: with ps; [
          numpy
          pip
          polars
          pytorch-lightning
          torch
          wandb
        ]))
      ];
    };
 };
}
