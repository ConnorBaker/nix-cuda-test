{
  lib,
  writeShellApplication,
  # packages
  python3,
  nixGL ? null,
  # Config
  wrapWithNixGL ? false,
}:
assert wrapWithNixGL -> nixGL != null; let
  optionalNixGLWrapper = lib.optionalString wrapWithNixGL "${nixGL.nixGLNvidia.name} ";
in
  writeShellApplication {
    name =
      "torch-cuda-is-available"
      + lib.optionalString wrapWithNixGL "-nixGL"
      + "-${python3.pkgs.torch.version}";
    runtimeInputs =
      [(python3.withPackages (ps: with ps; [torch]))]
      ++ lib.optionals wrapWithNixGL [nixGL.nixGLNvidia];
    text = ''
      ${optionalNixGLWrapper}python3 -c 'import torch; print(f"{torch.cuda.is_available()}")'
    '';
  }
