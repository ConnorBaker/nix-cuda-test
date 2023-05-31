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

        # Make an alias for python so it's wrapped with nixGLNvidia.
        shellHook = ''
          alias python3="${nixGL.nixGLNvidia.name} python3"
          alias python="${nixGL.nixGLNvidia.name} python3"
        '';
      };
  };
}
