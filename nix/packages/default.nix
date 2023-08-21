{
  perSystem = {
    config,
    pkgs,
    ...
  }: {
    packages = {
      nccl-test-suite = pkgs.callPackage ./nccl-test-suite.nix {};
      nccl-test-suite-nixGL = config.packages.nccl-test-suite.override {wrapWithNixGL = true;};
      nix-cuda-test = pkgs.python3Packages.callPackage ./nix-cuda-test.nix {};
      nix-cuda-test-nixGL = config.packages.nix-cuda-test.override {wrapWithNixGL = true;};
      torch-cuda-is-available = pkgs.callPackage ./torch-cuda-is-available.nix {};
      torch-cuda-is-available-nixGL = config.packages.torch-cuda-is-available.override {wrapWithNixGL = true;};
    };
  };
}
