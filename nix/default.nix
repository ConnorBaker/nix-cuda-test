{ inputs, ... }:
{
  imports = [
    ./options.nix
    ./apps
    ./nixosModules
    ./packages
    ./devShells
  ];

  perSystem =
    { config, system, ... }:
    let
      cfg = config.nix-cuda-test;
    in
    {
      _module.args.pkgs = import inputs.nixpkgs {
        inherit system;
        config = {
          allowBroken = true;
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
              enableIntelX86Extensions = system == "x86_64-linux";
              enable32bits = false;
              nvidiaVersion = cfg.nvidia.driver.version;
              nvidiaHash = cfg.nvidia.driver.hash;
            };
          })
          # Set up Python
          (
            _: prev:
            let
              # Names for python versions don't use underscores or dots
              python3AttributeVersion = builtins.replaceStrings [ "." ] [ "" ] cfg.python.version;
              python3 = prev."python${python3AttributeVersion}".override {
                enableOptimizations = cfg.python.optimize;
                self = python3;
              };
            in
            {
              # Use the optimized python build
              inherit python3;
            }
          )
          # TODO: Upstream this.
          # torchmetrics requires lightning utilities in newer versions.
          # https://github.com/Lightning-AI/torchmetrics/blob/7e9b18f58213c5cacbac4c66f09da71b3f233c55/requirements/base.txt#L9
          (_: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (pythonFinal: pythonPrev: {
                torchmetrics = pythonPrev.torchmetrics.overridePythonAttrs (oldAttrs: {
                  propagatedBuildInputs = oldAttrs.propagatedBuildInputs ++ [ pythonFinal.lightning-utilities ];
                });
              })
            ];
          })
          # Change the default version of CUDA used, wrap backendStdenv and cuda_nvcc in ccache
          (
            _: prev:
            let
              cudaPackagesAttributeVersion = builtins.replaceStrings [ "." ] [ "_" ] cfg.cuda.version;
            in
            {
              cudaPackages =
                if cfg.cuda.version != null then
                  prev."cudaPackages_${cudaPackagesAttributeVersion}"
                else
                  prev.cudaPackages;
            }
          )
        ];
      };
    };
}
