{
  config,
  inputs,
  withSystem,
  ...
}: {
  flake.nixosConfigurations.azure = withSystem "x86_64-linux" (
    {system, ...}:
      inputs.nixos-generators.nixosGenerate {
        inherit system;
        format = "azure";
        modules = [config.flake.nixosModules.azure];
      }
  );
}
