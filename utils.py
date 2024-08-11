import numpy as np
import torch

def preprocess_observation(obs):
    # Flatten (27, 84, 84, 3) feature maps into (81, 84, 84)
    combined_obs = np.concatenate([obs[:, :, :, i] for i in range(obs.shape[-1])], axis=0)
    return combined_obs

if __name__ == "__main__":
    zeros = np.zeros((27, 84, 84, 3))
    print(preprocess_observation(zeros).shape)