# nix_cuda_support

A project for testing Nix GPU support for machine learning workloads.

## Create Azure remote builder with Bicep

```bash
python3 ./azure/main.py \
  --action setup-vm \
  --location eastus \
  --resource-group NixOS \
  --storage-account nixosvhds \
  --storage-container vhds \
  --vhd ./svy64smy3qanrkr9l4iqylcfcx7v31jn-azure-image.vhd
```

TODO:

- [ ] Bash driver script to create resource group if non-existent, create storage account and upload VHD, and start VM(s).
- [ ] Try to build for each CUDA version
- [ ] Try to build for each supported CUDA capability

Debugging failure for PyTorch collect environment to run on NixOS without error.

Looks like it's missing `libnvidia-ml.so.1` compared to successful runs with NixGL.

August 21: master

```bash
nix run .#torch-cuda-is-available
```

```log
Python version: 3.10.12 (main, Jun  6 2023, 22:43:10) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-6.4.9-x86_64-with-glibc2.37
Is CUDA available: False
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 535.86.05
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True
```

May 25: python3Packages.torch: update CUDA capabilities for v2.0.1 release

```bash
nix run .#torch-cuda-is-available --override-input nixpkgs github:nixos/nixpkgs/a52e068d86845f7d182fe86bf9f1123817c0aea2
```

```log

```

April 10: python3Packages.pytorch: repair for darwin

```bash
nix run .#torch-cuda-is-available --override-input nixpkgs github:nixos/nixpkgs/0da597302cd18b88e9f6f8242a2c9dc6fa9891a5
```

```log

```