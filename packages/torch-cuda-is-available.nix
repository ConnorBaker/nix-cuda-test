{
  python3,
  writeShellApplication,
}:
writeShellApplication {
  name = "torch-cuda-is-available-${python3.pkgs.torch.version}";
  runtimeInputs = [ (python3.withPackages (ps: with ps; [ torch ])) ];
  text = ''
    python3 -c 'import torch; print(f"{torch.cuda.is_available()}")'
  '';
}
