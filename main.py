import torch
import torch.nn as nn

# Define the model architecture (placeholder)
class SimpleLandmarkModel(nn.Module):
    def __init__(self):
        super(SimpleLandmarkModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(136, 128),  # Adjust based on your actual model
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 136)
        )

    def forward(self, x):
        return self.fc_layers(x)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleLandmarkModel().to(device)

# Load model state_dict
model_path = "data/models/custom_landmark_model.pth"
state_dict = torch.load(model_path)

# Print model architecture
print("Model architecture:")
print(model)

# Print the keys in the state_dict
print("\nKeys in the model's state_dict:")
for key in state_dict.keys():
    print(key)

# Load the state_dict into the model and print the shape of each layer's weights
model.load_state_dict(state_dict)
print("\nLayer weight shapes:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data.shape}")

# Simulate a forward pass with dummy input to print layer output shapes
dummy_input = torch.randn(1, 136).to(device)  # Adjust this based on your model's expected input
print("\nForward pass output shapes:")
for layer in model.fc_layers:
    dummy_input = layer(dummy_input)
    print(f"After {layer}: {dummy_input.shape}")
