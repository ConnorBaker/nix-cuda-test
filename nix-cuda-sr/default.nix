{
  buildPythonPackage,
  config,
  einops,
  flash-attn,
  flit-core,
  lib,
  makeWrapper,
  optuna,
  pillow,
  pydantic,
  pyright,
  pytorch-lightning,
  rawpy,
  ruff,
  torch,
  torchmetrics,
  torchvision,
  transformer-engine,
  triton,
}:
let
  inherit (lib.fileset) toSource unions;
  inherit (lib.trivial) importTOML;

  pyprojectAttrs = importTOML ./pyproject.toml;

  finalAttrs = {
    # Must opt-out of __structuredAttrs which is on by default in our stdenv, but currently incompatible with Python
    # packaging: https://github.com/NixOS/nixpkgs/pull/347194.
    __structuredAttrs = false;

    pname = pyprojectAttrs.project.name;
    inherit (pyprojectAttrs.project) version;

    src = toSource {
      root = ./.;
      fileset = unions [
        ./pyproject.toml
        ./nix_cuda_sr
      ];
    };

    pyproject = true;

    build-system = [ flit-core ];

    nativeBuildInputs = [ makeWrapper ];

    dependencies = [
      flash-attn
      pydantic
      pytorch-lightning
      einops
      optuna
      pillow
      rawpy
      torch
      torchmetrics
      torchvision
      transformer-engine
      triton
    ];

    pythonImportsCheck = [ "nix_cuda_sr" ];

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
