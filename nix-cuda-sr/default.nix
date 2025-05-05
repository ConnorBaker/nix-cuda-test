{
  buildPythonPackage,
  config,
  cudaPackages,
  cutlass,
  einops,
  flash-attn,
  flit-core,
  lib,
  makeWrapper,
  onnx,
  optuna,
  pillow,
  pydantic,
  pyright,
  pytorch-lightning,
  rawpy,
  ruff,
  schedulefree,
  tensorboard-plugin-profile,
  tensorrt,
  modelopt,
  torch,
  torch-tb-profiler,
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
      cutlass
      einops
      flash-attn
      modelopt
      onnx
      optuna
      pillow
      pydantic
      pytorch-lightning
      rawpy
      schedulefree
      tensorboard-plugin-profile
      tensorrt
      torch
      torch-tb-profiler
      torchmetrics
      torchvision
      transformer-engine
      triton
    ] ++ modelopt.passthru.optional-dependencies.torch;

    pythonImportsCheck = [ "nix_cuda_sr" ];

    passthru.optional-dependencies.dev = [
      pyright
      ruff
    ];

    # TODO: Likely need a newer version of PyTorch to get this working.
    # torch._inductor.exc.LoweringException: AttributeError: 'CUTLASSArgs' object has no attribute 'exclude_kernels'
    # shellHook = ''
    #   export TORCHINDUCTOR_CUTLASS_DIR="${cudaPackages.cutlass.src}"
    # '';

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
