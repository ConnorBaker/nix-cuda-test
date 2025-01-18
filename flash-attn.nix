{
  buildPythonPackage,
  cmake,
  config,
  cudaPackages,
  einops,
  fetchFromGitHub,
  lib,
  ninja,
  psutil,
  setuptools,
  torch,
  wheel,
}:
let
  inherit (cudaPackages)
    cuda_cudart
    cuda_nvcc
    cutlass
    flags
    ;
  inherit (lib.attrsets) getOutput;
  inherit (lib.lists) any;
  inherit (lib.strings) concatStringsSep versionOlder;

  finalAttrs = {
    # Must opt-out of __structuredAttrs which is on by default in our stdenv, but currently incompatible with Python
    # packaging: https://github.com/NixOS/nixpkgs/pull/347194.
    __structuredAttrs = false;

    pname = "flash_attn";
    version = "2.7.2";

    src = fetchFromGitHub {
      owner = "Dao-AILab";
      repo = "flash-attention";
      tag = "v2.7.2.post1";
      hash = "sha256-jkBfOPXffQmi7pyBO6Qysx3kSOB9lYQI6e4IgGihJR8=";
    };

    pyproject = true;

    postPatch =
      # Bad
      ''
        substituteInPlace setup.py \
          --replace-fail \
            'subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])' \
            'pass' \
          --replace-fail \
            '+ cc_flag' \
            '+ ["${concatStringsSep ''","'' flags.gencode}"]'
        mkdir -p csrc/cutlass
        cp -r "${getOutput "include" cutlass}"/include csrc/cutlass/include
      '';

    # With 32 jobs, uses all 96 GB of RAM available on the machine; only succeeds with ZRAM enabled.
    # With default ZSTD compression, ZRAM reports less than 23 GB used to store ~170 GB of data.
    preConfigure = ''
      export BUILD_TARGET=cuda
      export FORCE_BUILD=TRUE
      export MAX_JOBS=$NIX_BUILD_CORES
    '';

    enableParallelBuilding = true;

    build-system = [
      cmake
      ninja
      psutil
      setuptools
      wheel
    ];

    nativeBuildInputs = [
      cuda_nvcc
    ];

    dontUseCmakeConfigure = true;

    dependencies = [
      einops
      torch
    ];

    buildInputs = [
      cuda_cudart
    ];

    doCheck = false;

    meta = with lib; {
      description = "Fast and memory-efficient exact attention";
      homepage = "https://github.com/Dao-AILab/flash-attention";
      license = licenses.bsd3;
      platforms = platforms.linux;
      maintainers = with maintainers; [ connorbaker ];
      broken =
        !config.cudaSupport || any (capability: versionOlder capability "8.0") flags.cudaCapabilities;
    };
  };
in
buildPythonPackage finalAttrs
