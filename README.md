# nix_cuda_support

A project for testing Nix GPU support for machine learning workloads.

- `nix run .#nccl-test-suite`
- `nix run .#torch-cuda-is-available`
- `nix run .#nix-cuda-test`

Optionally, to test a PR, use `--override-input nixpkgs github:nixos/nixpkgs/<commit>` with the appropriate commit from the PR.
