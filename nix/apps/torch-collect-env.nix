{
  nixGL,
  python3,
  writeShellApplication,
}:
writeShellApplication {
  name = "torch-collect-env";
  runtimeInputs = [
    (python3.withPackages (ps: with ps; [pip torch]))
    nixGL.nixGLNvidia
  ];
  text = ''
    ${nixGL.nixGLNvidia.name} python3 -m torch.utils.collect_env
  '';
}
