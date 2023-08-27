{
  lib,
  writeShellApplication,
  # packages
  cudaPackages,
  nixGL ? null,
  # Config
  wrapWithNixGL ? false,
}:
# Based off `nccl-tests`
assert wrapWithNixGL -> nixGL != null; let
  optionalNixGLWrapper = lib.optionalString wrapWithNixGL "${nixGL.nixGLNvidia.name} ";
in
  writeShellApplication {
    name =
      "nccl-test-suite"
      + lib.optionalString wrapWithNixGL "-nixGL"
      + "-${cudaPackages.nccl-tests.version}";
    runtimeInputs =
      [cudaPackages.nccl-tests]
      ++ lib.optionals wrapWithNixGL [nixGL.nixGLNvidia];
    text = ''
      for exe in ${cudaPackages.nccl-tests}/bin/*; do
        if [[ -x "$exe" && -f "$exe" ]]; then
          echo "Running $exe"
          ${optionalNixGLWrapper}"$exe"
        fi
      done
    '';
  }
