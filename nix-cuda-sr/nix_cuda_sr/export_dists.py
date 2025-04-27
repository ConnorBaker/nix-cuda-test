from pathlib import Path

import torch
from torchmetrics.functional.image.dists import DISTSNetwork

if __name__ == "__main__":
    # Example usage
    dists_network = DISTSNetwork()
    print(dists_network)
    example_inputs = (torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))
    onnx_program = torch.onnx.export(dists_network, example_inputs, Path("dists.onnx"), dynamo=False)
