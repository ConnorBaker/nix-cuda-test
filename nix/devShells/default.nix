{
  perSystem =
    {config, pkgs, ...}:
    {
      devShells = {
        nix-cuda-test = pkgs.mkShell {
          strictDeps = true;
          inputsFrom = [config.packages.nix-cuda-test];
          packages = config.packages.nix-cuda-test.optional-dependencies.dev;
        };
      };
    };
}
