{inputs, ...}: let
  mkApp = drv: inputs.flake-utils.lib.mkApp {inherit drv;};
in {
  perSystem = {packages, ...}: {
    apps = {
    };
  };
}
