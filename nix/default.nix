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
  }: {
    _module.args.pkgs = import inputs.nixpkgs {
      inherit system;
      overlays = [
        # Wrapper for nixGL
        (_: prev: {
          nixGL = import inputs.nixGL {
            pkgs = prev;
            enableIntelX86Extensions = true;
            enable32bits = false;
            nvidiaVersion = config.nvidia.driver.version;
            nvidiaHash = config.nvidia.driver.hash;
          };
        })
        # Change the default version of CUDA used
        (_: prev: {
          cudaPackages = prev.${config.cuda.packages};
        })
      ];
      config = {
        allowUnfree = true;
        cudaCapabilities = config.cuda.capabilities;
        cudaForwardCompat = config.cuda.forwardCompat;
        cudaSupport = config.cuda.support;
      };
    };
  };
}
