{
  python3,
  writeShellApplication,
}:
writeShellApplication {
  name = "xformers-info";
  runtimeInputs = [
    (python3.withPackages (ps: with ps; [ xformers ]))
  ];
  text = ''
    python3 -m xformers.info
  '';
}
