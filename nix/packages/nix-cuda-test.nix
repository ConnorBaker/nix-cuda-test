{
  buildPythonPackage,
  config,
  lib,
  # buildInputs
  cudaPackages,
  nixGL ? null,
  # propagatedBuildInputs
  click,
  openai-triton,
  pydantic,
  pytorch-lightning,
  stdenv,
  torch,
  torchvision,
  # passthru.optional-dependencies.dev
  black,
  mypy,
  pyright,
  ruff,
  # Config
  wrapWithNixGL ? false,
}:
assert wrapWithNixGL -> nixGL != null; let
  attrs = {
    pname = "nix-cuda-test" + lib.optionalString wrapWithNixGL "-nixGL";
    version = "0.1.0";
    format = "flit";
    src = lib.sources.sourceByRegex ../.. [
      "nix_cuda_test(:?/.*)?"
      "pyproject.toml"
    ];
    buildInputs = lib.optionals wrapWithNixGL [nixGL.nixGLNvidia];
    propagatedBuildInputs = [
      click
      openai-triton
      pydantic
      pytorch-lightning
      stdenv.cc # When building with openai-triton, we need a CPP compiler
      torch
      torchvision
    ];
    pythonImportsCheck = [
      "click"
      "nix_cuda_test"
      "pydantic"
      "pytorch_lightning"
      "torch"
      "torchvision"
    ];
    passthru.optional-dependencies.dev = [
      black
      mypy
      pyright
      ruff
    ];
    postInstall = lib.optionalString wrapWithNixGL ''
      mv $out/bin/nix-cuda-test $out/bin/.nix-cuda-test-wrapped
      echo '#!${stdenv.shell}' > $out/bin/nix-cuda-test
      echo '${nixGL.nixGLNvidia.name} $out/bin/.nix-cuda-test-wrapped "$@"' >> $out/bin/nix-cuda-test
    '';
    meta = with lib; {
      description = "A test of CUDA with nixpkgs";
      homepage = "";
      license = licenses.bsd3;
      platforms = platforms.linux;
      maintainers = with maintainers; [connorbaker];
      broken = !config.cudaSupport;
    };
  };
in
  buildPythonPackage attrs
