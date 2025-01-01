{
  config,
  cudaStdenv,
  lib,
  makeWrapper,
  pyright,
  python3,
  ruff,
  stdenv,
}:
let
  inherit (lib.fileset) toSource unions;
  inherit (lib.trivial) importTOML;
  inherit (python3.pkgs)
    buildPythonPackage
    flash-attn
    flit-core
    openai-triton
    pydantic
    pytorch-lightning
    torch
    torchvision
    transformer-engine
    ;

  pyprojectAttrs = importTOML ./pyproject.toml;

  finalAttrs = {
    # Must opt-out of __structuredAttrs which is on by default in our stdenv, but currently incompatible with Python
    # packaging: https://github.com/NixOS/nixpkgs/pull/347194.
    __structuredAttrs = false;
    stdenv = cudaStdenv;

    pname = pyprojectAttrs.project.name;
    inherit (pyprojectAttrs.project) version;

    src = toSource {
      root = ./.;
      fileset = unions [
        ./pyproject.toml
        ./nix_cuda_test
      ];
    };

    pyproject = true;

    build-system = [ flit-core ];

    nativeBuildInputs = [ makeWrapper ];

    dependencies = [
      flash-attn
      openai-triton # TODO: PyTorch should propagate this
      pydantic
      pytorch-lightning
      stdenv.cc # When building with openai-triton, we need a CPP compiler
      torch
      torchvision
      transformer-engine
    ];

    pythonImportsCheck = [
      "flash_attn"
      "nix_cuda_test"
      "pydantic"
      "pytorch_lightning"
      "torch"
      "torchvision"
      "transformer_engine"
    ];

    passthru.optional-dependencies.dev = [
      pyright
      ruff
    ];

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
buildPythonPackage finalAttrs
