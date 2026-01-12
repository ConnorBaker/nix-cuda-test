{
  buildPythonPackage,
  config,
  cutlass,
  einops,
  flash-attn,
  lib,
  makeWrapper,
  optuna,
  pillow,
  pydantic,
  pyright,
  pytorch-lightning,
  ruff,
  setuptools,
  tensorboard-plugin-profile,
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
    __structuredAttrs = true;

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

    build-system = [ setuptools ];

    nativeBuildInputs = [ makeWrapper ];

    dependencies = [
      cutlass
      einops
      flash-attn
      optuna
      pillow
      pydantic
      pytorch-lightning
      tensorboard-plugin-profile
      torch
      torch-tb-profiler
      torchmetrics
      torchvision
      transformer-engine
      triton
    ];

    pythonImportsCheck = [ "nix_cuda_sr" ];

    optional-dependencies.dev = [
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
