import torch
from diffusion_common import ResUNet

def main():
    print("Hello from ddpm!")
    
    input_dim = 3
    output_dim = 3
    layer_dims = [64, 128]
    bridge_dim = 256
    
    model = ResUNet(input_dim, output_dim, layer_dims, bridge_dim)
    print(f"Successfully initialized ResUNet with {sum(p.numel() for p in model.parameters())} parameters.")


if __name__ == "__main__":
    main()
