{
  config,
  cudaStdenv,
  lib,
  makeWrapper,
  pyright,
  python3,
  ruff,
}:
let
  inherit (lib.fileset) toSource unions;
  inherit (lib.trivial) importTOML;
  inherit (python3.pkgs)
    buildPythonPackage
    flash-attn
    flit-core
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
      pydantic
      pytorch-lightning
      torch
      torchvision
      transformer-engine
    ];

    pythonImportsCheck = [ "nix_cuda_test" ];

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
