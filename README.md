# nix_cuda_support

A project for testing Nix GPU support for machine learning workloads.

- `nix run .#nccl-test-suite`
- `nix run .#torch-cuda-is-available`
- `nix run .#nix-cuda-test`

Optionally, to test a PR, use `--override-input nixpkgs github:nixos/nixpkgs/<commit>` with the appropriate commit from the PR.

## To-do

- Add support for `treefmt`
- Matrix `cudaPackages` and `cudaCapabilities` to provide multiple variants of the same package
- Investigate the performance impact of using an optimized python build

    ```nix
    (_: prev: {
      python3 = prev.python3.override {
        enableOptimizations = true;
        enableLTO = true;
        reproducibleBuild = false;
        self = python3;
      };
    })
    ```

- Investigate compile times as a result of using [`fastStdenv`](https://nixos.wiki/wiki/C#Faster_GCC_compiler)
- Investigate link times as a result of using [`useMoldLinker`](https://github.com/NixOS/nixpkgs/blob/dbb569b8539424ed7d757bc080adb902ba84a086/pkgs/stdenv/adapters.nix#L192)
- Investigate local builds using [`ccacheStdenv`](https://nixos.wiki/wiki/CCache)