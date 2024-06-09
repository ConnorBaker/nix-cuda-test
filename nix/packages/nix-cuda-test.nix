{
  buildPythonPackage,
  config,
  lib,
  # nativeBuildInputs
  flit-core,
  # buildInputs
  # cudaPackages,
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
assert wrapWithNixGL -> nixGL != null;
let
  attrs = {
    pname = "nix-cuda-test" + lib.optionalString wrapWithNixGL "-nixGL";
    version = "0.1.0";
    pyproject = true;
    src = lib.sources.sourceByRegex ../.. [
      "nix_cuda_test(:?/.*)?"
      "pyproject.toml"
    ];
    build-system = [ flit-core ];
    buildInputs = lib.optionals wrapWithNixGL [ nixGL.nixGLNvidia ];
    dependencies = [
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
      mv "$out/bin/nix-cuda-test" "$out/bin/.nix-cuda-test-wrapped"
      echo '#!${stdenv.shell}' > "$out/bin/${attrs.pname}"
      echo '"${lib.getExe nixGL.nixGLNvidia}"'\
        " \"$out/bin/.nix-cuda-test-wrapped\""\
        '"$@"' \
        >> "$out/bin/${attrs.pname}"
      chmod +x "$out/bin/${attrs.pname}"
    '';
    meta = with lib; {
      description = "A test of CUDA with nixpkgs";
      homepage = "";
      license = licenses.bsd3;
      platforms = platforms.linux;
      maintainers = with maintainers; [ connorbaker ];
      broken = !config.cudaSupport;
    };
  };
in
buildPythonPackage attrs
