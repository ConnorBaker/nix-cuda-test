{
  perSystem = {pkgs, ...}: {
    devShells.default = with pkgs;
      mkShell {
        packages = [
          nixGL.nixGLNvidia
          ruff
          (python3.withPackages (ps:
            with ps; [
              openai-triton
              pytorch-lightning
              torch
              torchvision
            ]))
        ];
      };
  };
}
