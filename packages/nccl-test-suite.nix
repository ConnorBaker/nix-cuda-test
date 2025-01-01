{
  cudaPackages,
  lib,
  writeShellApplication,
}:
let
  inherit (lib.attrsets) getBin;
in
# Based off `nccl-tests`
writeShellApplication {
  name = "nccl-test-suite-${cudaPackages.nccl-tests.version}";
  runtimeInputs = [ cudaPackages.nccl-tests ];
  text = ''
    for exe in "${getBin cudaPackages.nccl-tests}/bin/"*; do
      if [[ -x "$exe" && -f "$exe" ]]; then
        echo "Running $exe"
        "$exe"
      fi
    done
  '';
}
