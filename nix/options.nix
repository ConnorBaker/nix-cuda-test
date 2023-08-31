{lib, ...}: let
  inherit (lib) mkOption;
  inherit (lib.types) bool nonEmptyListOf str submodule;
in {
  perSystem.options.nix-cuda-test = {
    cuda = mkOption {
      description = "CUDA options";
      type = submodule {
        options = {
          capabilities = mkOption {
            description = "List of CUDA capabilities to build for";
            example = ["3.5" "5.2" "6.1" "7.0" "7.5" "8.0" "8.6"];
            type = nonEmptyListOf str;
          };
          forwardCompat = mkOption {
            description = "Whether to build for forward compatibility";
            type = bool;
          };
          version = mkOption {
            description = "CUDA version to use";
            example = "11.8";
            type = str;
          };
        };
      };
    };
    nvidia.driver = mkOption {
      description = "NVIDIA driver options";
      type = submodule {
        options = {
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
    };
    python = mkOption {
      description = "Python options";
      type = submodule {
        options = {
          version = mkOption {
            description = "Python version to build use";
            example = "3.8";
            type = str;
          };
          optimize = mkOption {
            description = "Whether to build with optimizations";
            type = bool;
          };
        };
      };
    };
  };
}
