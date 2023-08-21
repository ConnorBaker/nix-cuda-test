{lib, ...}: let
  inherit (lib) mkOption;
  inherit (lib.types) bool nonEmptyListOf str;
in {
  perSystem.options = {
    cuda = {
      capabilities = mkOption {
        description = "List of CUDA capabilities to build for";
        example = ["3.5" "5.2" "6.1" "7.0" "7.5" "8.0" "8.6"];
        type = nonEmptyListOf str;
      };
      forwardCompat = mkOption {
        description = "Whether to build for forward compatibility";
        type = bool;
      };
      packages = mkOption {
        description = "Attribute name of the CUDA package set to use";
        example = "cudaPackages_11_8";
        type = str;
      };
      support = mkOption {
        description = "Whether to build for support CUDA libraries";
        type = bool;
      };
    };
    nvidia.driver = {
      version = mkOption {
        description = "NVIDIA driver version to build against";
        example = "535.86.05";
        type = str;
      };
      hash = mkOption {
        description = "NVIDIA driver hash to build against";
        example = "sha256-QH3wyjZjLr2Fj8YtpbixJP/DvM7VAzgXusnCcaI69ts=";
        type = str;
      };
    };
  };
}
