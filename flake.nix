{
  inputs = {
    cuda-packages.url = "github:ConnorBaker/cuda-packages";
    flake-parts.follows = "cuda-packages/flake-parts";
    nixpkgs.follows = "cuda-packages/nixpkgs";
    git-hooks-nix.follows = "cuda-packages/git-hooks-nix";
    treefmt-nix.follows = "cuda-packages/treefmt-nix";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];

      imports = [
        inputs.treefmt-nix.flakeModule
        inputs.git-hooks-nix.flakeModule
      ];

      flake.overlays.default = import ./overlay.nix;

      perSystem =
        {
          config,
          pkgs,
          system,
          ...
        }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            # TODO: Due to the way Nixpkgs is built in stages, the config attribute set is not re-evaluated.
            # This is problematic for us because we use it to signal the CUDA capabilities to the overlay.
            # The only way I've found to combat this is to use pkgs.extend, which is not ideal.
            # TODO: This also means that Nixpkgs needs to be imported *with* the correct config attribute set
            # from the start, unless they're willing to re-import Nixpkgs with the correct config.
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
            overlays = [
              inputs.cuda-packages.overlays.default
              inputs.self.overlays.default
            ];
          };

          legacyPackages = pkgs;

          packages = {
            inherit (pkgs.pkgsCuda.sm_89.cudaPackages.tests)
              nccl-test-suite
              nix-cuda-test
              torch-cuda-is-available
              xformers-info
              ;
          };

          devShells =
            let
              inherit (pkgs.pkgsCuda.sm_89.cudaPackages.tests) nix-cuda-test;
            in
            {
              # default = config.treefmt.build.devShell;
              default = pkgs.pkgsCuda.sm_89.mkShell {
                strictDeps = true;
                inputsFrom = [ nix-cuda-test ];
                packages = nix-cuda-test.optional-dependencies.dev;
              };
            };

          pre-commit.settings.hooks = {
            # Formatter checks
            treefmt = {
              enable = true;
              package = config.treefmt.build.wrapper;
            };

            # Nix checks
            deadnix.enable = true;
            nil.enable = true;
            statix.enable = true;
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              # Markdown, YAML, JSON
              prettier = {
                enable = true;
                includes = [
                  "*.json"
                  "*.md"
                  "*.yaml"
                ];
                settings = {
                  embeddedLanguageFormatting = "auto";
                  printWidth = 120;
                  tabWidth = 2;
                };
              };

              # Nix
              nixfmt.enable = true;

              # Python
              ruff.enable = true;

              # Shell
              shellcheck.enable = true;
              shfmt.enable = true;
            };
          };
        };
    };
}
