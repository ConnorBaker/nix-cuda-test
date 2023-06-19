{
  inputs = {
    flake-parts = {
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
    extra-trusted-substituters = [
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
          cudaCapabilities = ["8.0" "8.6" "8.9"];
          cudaPackages = "cudaPackages_11_8";
          cudaForwardCompat = true;
          formatter = pkgs.alejandra;
        };
      };
    };
}
