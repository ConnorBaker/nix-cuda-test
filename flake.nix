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
      perSystem = {
        config,
        pkgs,
        ...
      }: {
        config = {
          cuda = {
            capabilities = ["8.9"];
            packages = "cudaPackages_11_8";
            forwardCompat = false;
            support = true;
          };
          formatter = pkgs.alejandra;
          nvidia.driver = {
            hash = "sha256-QH3wyjZjLr2Fj8YtpbixJP/DvM7VAzgXusnCcaI69ts=";
            version = "535.86.05";
          };
          pre-commit.settings = {
            hooks = {
              # Nix checks
              alejandra.enable = true;
              deadnix.enable = true;
              nil.enable = true;
              statix.enable = true;
              # Python checks
              black.enable = true;
              mypy.enable = true;
              pyright.enable = true;
              ruff.enable = true;
            };
            settings = let
              # We need to provide wrapped version of mypy and pyright which can find our imports.
              wrapper = name:
                pkgs.writeShellScript name ''
                  source ${config.devShells.nix-cuda-test}
                  ${name} "$@"
                '';
            in {
              mypy.binPath = "${wrapper "mypy"}";
              pyright.binPath = "${wrapper "pyright"}";
            };
          };
        };
      };
    };
}
