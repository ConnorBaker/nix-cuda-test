final:
let
  inherit (final.lib.attrsets) recurseIntoAttrs recursiveUpdate;
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
  magma = prev.magma.override (prevAttrs: {
    cudaPackages = recursiveUpdate prevAttrs.cudaPackages {
      flags.dropDot = prevAttrs.cudaPackages.flags.dropDots;
    };
  });
  pythonPackagesExtensions = prev.pythonPackagesExtensions or [ ] ++ [
    (finalPythonPackages: prevPythonPackages: {
      # TODO: Upstream
      flash-attn = finalPythonPackages.callPackage ./flash-attn.nix { };
      # TODO: Upstream
      transformer-engine = finalPythonPackages.callPackage ./transformer-engine.nix { };
      torch =
        (prevPythonPackages.torch.override {
          # PyTorch doesn't need Triton to build.
          # Just include it in whichever package consumes pytorch.
          tritonSupport = false;
        }).overrideAttrs
          (prevAttrs: {
            buildInputs = prevAttrs.buildInputs or [ ] ++ [
              final.cudaPackages.nccl.static
            ];
          });
      triton = prevPythonPackages.triton.overrideAttrs (
        let
          inherit (final.stdenv) cc;
        in
        finalAttrs: prevAttrs: {
          env = prevAttrs.env or { } // {
            CC = "${cc}/bin/${cc.targetPrefix}cc";
            CXX = "${cc}/bin/${cc.targetPrefix}c++";
          };
          preConfigure =
            prevAttrs.preConfigure or ""
            # Patch in our compiler.
            # https://github.com/triton-lang/triton/blob/cf34004b8a67d290a962da166f5aa2fc66751326/python/triton/runtime/build.py#L25
            + ''
              substituteInPlace "$NIX_BUILD_TOP/$sourceRoot/python/triton/runtime/build.py" \
                --replace-fail \
                  'cc = os.environ.get("CC")' \
                  'cc = "${finalAttrs.env.CC}"'
            '';
        }
      );
    })
  ];
}
