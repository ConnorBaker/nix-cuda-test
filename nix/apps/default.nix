{inputs, ...}: let
  mkApp = drv: inputs.flake-utils.lib.mkApp {inherit drv;};
in {
  perSystem = {pkgs, ...}: {
    apps = {
      torch-collect-env = mkApp (pkgs.callPackage ./torch-collect-env.nix {});
      # xformers-info = mkApp (pkgs.callPackage ./xformers-info.nix {});
    };
  };
}
