{
  lib,
  writeShellApplication,
  # packages
  python3,
  nixGL ? null,
  # Config
  wrapWithNixGL ? false,
}:
assert wrapWithNixGL -> nixGL != null;
  writeShellApplication {
    name = "torch-collect-env" + lib.optionalString wrapWithNixGL "-nixGL";
    runtimeInputs =
      [(python3.withPackages (ps: with ps; [pip torch]))]
      ++ lib.optionals wrapWithNixGL [nixGL.nixGLNvidia];
    text = builtins.concatStringsSep " " (
      lib.optionals wrapWithNixGL [nixGL.nixGLNvidia.name]
      ++ ["python3 -m torch.utils.collect_env"]
    );
  }
