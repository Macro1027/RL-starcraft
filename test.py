import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(observation_space[0], 64, kernel_size=(5, 5)),
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
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(256 * 2 * 2, 256),  # Adjust based on calculated flattened size
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
        x = x.unsqueeze(0)  # Add batch dimension
        x, _ = self.lstm(x)
        x = self.linear_softmax(x)
        return x

# Define the environment
env_id = 'CartPole-v1'  # Example environment
env = make_vec_env(env_id, n_envs=1)

# Define the model
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(in_shape=(3, 32, 32), out_shape=env.action_space.n)
)

model = DQN('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./dqn_tensorboard/")

# Create an evaluation callback
eval_env = make_vec_env(env_id, n_envs=1)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

# Train the model
model.learn(total_timesteps=10000, callback=eval_callback)

# # Save the model
# model.save("dqn_cartpole")

# # Load the model
# model = DQN.load("dqn_cartpole")

# # Evaluate the model
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} +/- {std_reward}")
