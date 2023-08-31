{
  perSystem = {
    lib,
    pkgs,
    ...
  }: let
    ourPackages = {
      nccl-test-suite = pkgs.callPackage ./nccl-test-suite.nix {};
      nix-cuda-test = pkgs.python3Packages.callPackage ./nix-cuda-test.nix {};
      torch-cuda-is-available = pkgs.callPackage ./torch-cuda-is-available.nix {};
    };
    wrapPackages = lib.mapAttrs' (name: value: {
      name = "${name}-nixGL";
      value = value.override {
        wrapWithNixGL = true;
      };
    });
  in {
    packages = ourPackages // wrapPackages ourPackages;
  };
}
