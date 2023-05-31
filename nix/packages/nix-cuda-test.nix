{
  buildPythonPackage,
  config,
  lib,
  # propagatedBuildInputs
  pip,
  pytorch-lightning,
  torch,
}: let
  attrs = {
    pname = "nix-cuda-test";
    version = "0.1.0";
    format = "pyproject";
    src = ../..;
    # Pip is required for torch.utils.collect_env to work.
    nativeBuildInputs = [pip];
    buildInputs = [pip];
    propagatedBuildInputs = [pip pytorch-lightning torch];
    pythonImportsCheck = [
      "nix_cuda_test"
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
