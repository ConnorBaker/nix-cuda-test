final: prev:
let
  inherit (prev.lib.attrsets) recurseIntoAttrs;
  inherit (prev.lib.lists) optionals;
in
{
  cudaPackagesExtensions = prev.cudaPackagesExtensions or [ ] ++ [
    (finalCudaPackages: prevCudaPackages: {
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
      torch =
        # Could not find CUPTI library, using CPU-only Kineto build
        # Could NOT find NCCL (missing: NCCL_INCLUDE_DIR)
        # USE_TENSORRT is unset in the printed config at the end of configurePhase.
        # Not sure if that's used directly or passed through to one of the vendored projects.
        (prevPythonPackages.torch.override (prevAttrs: {
          # PyTorch doesn't need Triton to build.
          # Just include it in whichever package consumes pytorch.
          tritonSupport = false;
        })).overrideAttrs
          (prevAttrs: {
            buildInputs =
              prevAttrs.buildInputs or [ ]
              ++ [
                final.cudaPackages.libcusparse_lt
                final.cudaPackages.libcudss
                final.cudaPackages.libcufile
              ]
              ++ optionals (final.cudaPackages.nccl.meta.available) [ final.cudaPackages.nccl.static ];

            USE_CUFILE = 1;
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
