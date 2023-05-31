{
  perSystem = {pkgs, ...}: {
    packages = {
      nix-cuda-test = pkgs.python3Packages.callPackage ./nix-cuda-test.nix {};
    };
  };
}
