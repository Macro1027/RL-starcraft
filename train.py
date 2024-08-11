from model import CustomCNN
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from sc2env import SC2GymEnv
from absl import app
from utils import preprocess_observation

def main(unused_args):
    gym.envs.registration.register(
        id='SC2Simple64',
        entry_point='__main__:SC2GymEnv',
    )

    # Create the environment
    env = gym.make('SC2Simple64', vis=False)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128)
    )

    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=1000)

if __name__ == "__main__":
    app.run(main)

