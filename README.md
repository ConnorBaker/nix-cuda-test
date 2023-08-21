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
