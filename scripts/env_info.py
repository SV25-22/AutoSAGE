import torch, platform, sys, os
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "torch.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
