{
  nixGL,
  python3,
  writeShellApplication,
}:
writeShellApplication {
  name = "xformers-info";
  runtimeInputs = [
    (python3.withPackages (ps: with ps; [ xformers ]))
    nixGL.nixGLNvidia
  ];
  text = ''
    ${nixGL.nixGLNvidia.name} python3 -m xformers.info
  '';
}
