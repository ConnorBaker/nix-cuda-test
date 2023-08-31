{inputs, ...}: {
  imports = [
    ./options.nix
    ./apps
    ./nixosModules
    ./packages
    ./devShells
  ];

  perSystem = {
    config,
    system,
    ...
  }: let
    cfg = config.nix-cuda-test;
  in {
    _module.args.pkgs = import inputs.nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaCapabilities = cfg.cuda.capabilities;
        cudaForwardCompat = cfg.cuda.forwardCompat;
        cudaSupport = true;
      };
      overlays = [
        # Wrapper for nixGL
        (_: prev: {
          nixGL = import inputs.nixGL {
            pkgs = prev;
            enableIntelX86Extensions = true;
            enable32bits = false;
            nvidiaVersion = cfg.nvidia.driver.version;
            nvidiaHash = cfg.nvidia.driver.hash;
          };
        })
        # Set up Python
        (_: prev: let
          # Names for python versions don't use underscores or dots
          python3AttributeVersion = builtins.replaceStrings ["."] [""] cfg.python.version;
          python3 = prev."python${python3AttributeVersion}".override {
            enableOptimizations = cfg.python.optimize;
            enableLTO = true;
            reproducibleBuild = false;
            self = python3;
          };
        in {
          # Use the optimized python build
          inherit python3;
        })
        # Change the default version of CUDA used, wrap backendStdenv and cuda_nvcc in ccache
        (_: prev: let
          cudaPackagesAttributeVersion = builtins.replaceStrings ["."] ["_"] cfg.cuda.version;
        in {
          cudaPackages = prev."cudaPackages_${cudaPackagesAttributeVersion}";
        })
      ];
    };
  };
}
