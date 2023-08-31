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
    pre-commit-hooks-nix = {
      inputs = {
        flake-utils.follows = "flake-utils";
        nixpkgs-stable.follows = "nixpkgs";
        nixpkgs.follows = "nixpkgs";
      };
      url = "github:cachix/pre-commit-hooks.nix";
    };
  };

  nixConfig = {
    # Add my own cache and the CUDA maintainer's cache
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      imports = [
        inputs.pre-commit-hooks-nix.flakeModule
        ./nix
      ];
      perSystem = {pkgs, ...}: {
        nix-cuda-test = {
          cuda = {
            capabilities = ["8.9"];
            version = "11.8";
            forwardCompat = false;
          };
          nvidia.driver = {
            hash = "sha256-L51gnR2ncL7udXY2Y1xG5+2CU63oh7h8elSC4z/L7ck=";
            version = "535.104.05";
          };
          python = {
            optimize = false;
            version = "3.10";
          };
        };
        formatter = pkgs.alejandra;
        pre-commit = {
          settings.hooks = {
            # Nix checks
            alejandra.enable = true;
            deadnix.enable = true;
            nil.enable = true;
            statix.enable = true;
            # Python checks
            black.enable = true;
            ruff.enable = true;
            # Python type checkers -- require access to the stubs the environment has.
            # Unsure how to supply them with those given that they're populated by different hooks
            # only run inside the environment.
            mypy.enable = false;
            pyright.enable = false;
          };
        };
      };
    };
}
