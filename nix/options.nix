{lib, ...}: let
  inherit (lib) mkOption;
  inherit (lib.types) bool nonEmptyListOf str;
in {
  perSystem.options = {
    cudaCapabilities = mkOption {
      description = "List of CUDA capabilities to build for";
      example = ["3.5" "5.2" "6.1" "7.0" "7.5" "8.0" "8.6"];
      type = nonEmptyListOf str;
    };
    cudaForwardCompat = mkOption {
      description = "Whether to build for forward compatibility";
      example = false;
      type = bool;
    };
    cudaPackages = mkOption {
      description = "Attribute name of the CUDA package set to use";
      example = "cudaPackages_11_8";
      type = str;
    };
  };
}
