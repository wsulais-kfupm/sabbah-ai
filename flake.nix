{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    # nixpkgs.url = "github:NixOS/nixpkgs/haskell-updates";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
      in {
        devShells.default = pkgs.mkShell {
          PYTORCH_ENABLE_MPS_FALLBACK = true;
          buildInputs = with pkgs; [
            # haskellPackages.pandoc_2_19_2
            pandoc
            black
            nodePackages.pyright
            python3Packages.ipython
            python3Packages.pip
            python3Packages.gensim
            python3Packages.numpy
            python3Packages.torch
            python3Packages.transformers
          ];
        };
      }
    );
}
