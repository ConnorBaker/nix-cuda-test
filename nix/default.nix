{inputs, ...}: {
  imports = [
    ./options.nix
    ./apps
    ./nixosConfigurations
    ./nixosModules
    ./packages
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
            nvidiaVersion = "535.54.03";
            nvidiaHash = "sha256-RUdk9X6hueGRZqNw94vhDnHwYmQ4+xl/cm3DyvBbQII=";
          };
        })
        # Change the default version of CUDA used
        (_: prev: {
          cudaPackages = prev.${config.cudaPackages};
        })
        # Use a newer version of Nix to take advantage of max-substitution-jobs
        (_: prev: {
          nix = let
            inherit (prev) nix;
            inherit (prev.nixVersions) nix_2_16;
            inherit (prev.lib.strings) versionAtLeast;
          in
            if versionAtLeast nix.version "2.16"
            then nix
            else nix_2_16;
        })
      ];
      config = {
        inherit (config) cudaCapabilities cudaForwardCompat;
        allowUnfree = true;
        cudaSupport = true;
      };
    };
  };
}
