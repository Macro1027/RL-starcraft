import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        # Calculate the output size after convolutional layers
        n_input_channels = observation_space.shape[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.3)
        )
        
        # Compute the shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            conv_output = self.conv3(self.conv2(self.conv1(sample_input)))
            n_flatten = conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3]
        
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.linear_softmax = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear_relu(x)
        x = x.unsqueeze(0)  # Add batch dimension for LSTM
        x, _ = self.lstm(x)
        x = x.squeeze(0)  # Remove batch dimension after LSTM
        x = self.linear_softmax(x)
        return x