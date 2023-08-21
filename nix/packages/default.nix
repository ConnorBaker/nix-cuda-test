{
  perSystem = {
    config,
    pkgs,
    ...
  }: {
    packages = {
      nix-cuda-test = pkgs.python3Packages.callPackage ./nix-cuda-test.nix {};
      nix-cuda-test-nixGL = config.packages.nix-cuda-test.override {wrapWithNixGL = true;};
      torch-collect-env = pkgs.callPackage ./torch-collect-env.nix {};
      torch-collect-env-nixGL = config.packages.torch-collect-env.override {wrapWithNixGL = true;};
    };
  };
}
