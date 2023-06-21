{
  buildPythonPackage,
  config,
  lib,
  # propagatedBuildInputs
  click,
  openai-triton,
  pydantic,
  pytorch-lightning,
  torch,
  torchvision,
}: let
  attrs = {
    pname = "nix-cuda-test";
    version = "0.1.0";
    format = "pyproject";
    src = lib.sources.sourceByRegex ../.. [
      "nix_cuda_test(:?/.*)?"
      "pyproject.toml"
    ];
    propagatedBuildInputs = [
      click
      openai-triton
      pydantic
      pytorch-lightning
      torch
      torchvision
    ];
    pythonImportsCheck = [
      "click"
      "nix_cuda_test"
      "pydantic"
      "pytorch_lightning"
      "torch"
    ];
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
