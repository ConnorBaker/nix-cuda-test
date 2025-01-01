final:
let
  inherit (final.lib.attrsets) getOutput recurseIntoAttrs;
in
prev: {
  cudaPackagesExtensions = prev.cudaPackagesExtensions or [ ] ++ [
    (finalCudaPackages: prevCudaPackages: {
      # TODO(@connorbaker): Note sure package sets are set up in such a way that each CUDA package set is its own
      # default. For example, will `pkgsCuda.sm_89.cudaPackages_12_2_2.pkgs.callPackage` provide `pkgsCuda.sm_89.cudaPackages_12_2_2`
      # as `cudaPackages`?
      tests = recurseIntoAttrs (prevCudaPackages.tests or { }) // {
        nix-cuda-test = finalCudaPackages.callPackage ./packages/nix-cuda-test { };
        nccl-test-suite = finalCudaPackages.callPackage ./packages/nccl-test-suite.nix { };
        torch-cuda-is-available = finalCudaPackages.callPackage ./packages/torch-cuda-is-available.nix { };
        xformers-info = finalCudaPackages.callPackage ./packages/xformers-info.nix { };
      };
    })
  ];
  pythonPackagesExtensions = prev.pythonPackagesExtensions or [ ] ++ [
    (finalPythonPackages: prevPythonPackages: {
      # TODO: Upstream
      flash-attn = finalPythonPackages.callPackage ./flash-attn.nix { };
      # TODO: Upstream
      transformer-engine = finalPythonPackages.callPackage ./transformer-engine.nix { };
      # TODO: Why overridePythonAttrs? Isn't that a footgun? Why does overrideAttrs not work?
      torch = prevPythonPackages.torch.overrideAttrs (prevAttrs: {
        buildInputs = prevAttrs.buildInputs or [ ] ++ [
          (getOutput "static" final.cudaPackages.nccl.static)
        ];
      });
      # TODO: Oh my god using overridePythonAttrs removes the `override` attribute?
      # When using that instead of overrideAttrs, I get this error:
      #        error: attribute 'override' missing
      #  at /nix/store/f0m7vb7sihbqhynbiscj6ngcjgsm9kqw-source/pkgs/top-level/python-packages.nix:16402:17:
      #   16401|
      #   16402|   triton-cuda = self.triton.override {
      #        |                 ^
      #   16403|     cudaSupport = true;
      # TODO: Upstream in progress: https://github.com/NixOS/nixpkgs/pull/369495
      triton = prevPythonPackages.triton.overrideAttrs (prevAttrs: {
        preConfigure =
          prevAttrs.preConfigure or ""
          # Patch the triton source to not use ldconfig.
          # https://github.com/triton-lang/triton/blob/2939d86fc5c4bbb64fd04fd5346a6dbed3bc3c85/third_party/nvidia/backend/driver.py#L27
          + ''
            substituteInPlace "$NIX_BUILD_TOP/$sourceRoot/third_party/nvidia/backend/driver.py" \
              --replace-fail \
                'libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()' \
                'libs = ""'
          ''
          # Patch the source code to make sure it doesn't specify a non-existent PTXAS version.
          # CUDA 12.6 (the current default/max) tops out at PTXAS version 8.5.
          # NOTE: This is fixed in `master`:
          # https://github.com/triton-lang/triton/commit/f48dbc1b106c93144c198fbf3c4f30b2aab9d242
          + ''
            substituteInPlace "$NIX_BUILD_TOP/$sourceRoot/third_party/nvidia/backend/compiler.py" \
              --replace-fail \
                'return 80 + minor' \
                'return 80 + (minor if minor <= 5 else 5)'
          ''
        # TODO: Had to use `export LD_LIBRARY_PATH=/run/opengl-driver/lib` to get it to detect libcuda.so.
        # The only reason that works:
        # https://github.com/triton-lang/triton/commit/0149bf70042efb998fc7a1ea30ed99e3a0f75053
        ;
      });
    })
  ];
}
