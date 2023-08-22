import torch
import torchvision.models as models
from models.resnet_pt import ResNet8

# Create an instance of ResNet8
params = {'in_channels': 3, 'out_channels': 10, 'activation': 'Default'}
model = ResNet8(params)

# Load the state dict from the .pt file
state_dict = torch.load('ResNet8.pt')
model.load_state_dict(state_dict)

# Set input and output names
input_names = ["input"]
output_names = ["output"]

# Set input shape
input_shape = (1, 3, 32, 32)
input = torch.randn(1,3,32,32)
# Set the model to evaluation mode
model.eval()

# Export the model to ONNX format
torch.onnx.export(model,
                input,
                "ResNet8.onnx",
                input_names = input_names,
                output_names = output_names,
                export_params=True,
)
