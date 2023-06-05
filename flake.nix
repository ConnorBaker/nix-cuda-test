{
  inputs = {
    flake-parts = {
      # TODO(@connorbaker): Using nixpkgs-lib fails with the following:
      #   … while evaluating the attribute 'flake-parts.lib.mkFlake'
      #     at /nix/store/jiyanhix45y0a7m9nsrng44yjzrp79m9-source/flake.nix:9:5:
      #       8|   outputs = { nixpkgs-lib, ... }: {
      #       9|     lib = import ./lib.nix {
      #        |     ^
      #      10|       inherit (nixpkgs-lib) lib;
      #   (stack trace truncated; use '--show-trace' to show the full trace)
      #   error: getting status of '/nix/store/dpj48n6j242fh3vq27fgwdcybaa7lydb-source/.version':
      #   No such file or directory
      inputs.nixpkgs-lib.follows = "nixpkgs";
      url = "github:hercules-ci/flake-parts";
    };
    flake-utils.url = "github:numtide/flake-utils";
    nixGL = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:guibou/nixGL";
    };
    nixos-generators = {
      inputs.nixlib.follows = "nixpkgs-lib";
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:nix-community/nixos-generators";
    };
    nixpkgs.url = "github:NixOS/nixpkgs";
    nixpkgs-lib.url = "github:nix-community/nixpkgs.lib";
  };

  nixConfig = {
    # Add my own cache and the CUDA maintainer's cache
    extra-substituters = [
      "https://cantcache.me"
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cantcache.me:Y+FHAKfx7S0pBkBMKpNMQtGKpILAfhmqUSnr5oNwNMs="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      imports = [./nix];
      perSystem = {pkgs, ...}: {
        config = {
          cudaCapabilities = ["8.9"];
          cudaPackages = "cudaPackages_11_8";
          cudaForwardCompat = false;
          formatter = pkgs.alejandra;
        };
      };
    };
}
