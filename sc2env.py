import gymnasium as gym
from gymnasium import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app
import numpy as np

class SC2GymEnv(gym.Env):
    def __init__(self, map_name="Simple64", vis=True):
        super(SC2GymEnv, self).__init__()
        self.env = sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=vis
        )
        self.action_space = spaces.Discrete(len(actions.FUNCTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    def reset(self, **kwargs):
        self.env.reset()
        obs = self.env.step([actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])])
        return self._process_obs(obs[0]), {}

    def step(self, action):
        obs = self.env.step([actions.FunctionCall(action, [])])
        reward = obs[0].reward
        done = obs[0].last()
        return self._process_obs(obs[0]), reward, done, {}

    def _process_obs(self, obs):
        screen = obs.observation["feature_screen"]
        screen = np.stack([screen] * 3, axis=-1) # Stack to create a 3-channel image

        return screen

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()

def main(unused_args):
    env = SC2GymEnv()
    try:
        env.reset()
        # Take a no-op action to render the environment
        while True:
            actions_list = [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]
            obs = env.step(actions_list)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

def test(unused_args):
    env = SC2GymEnv(vis=False)
    state, _ = env.reset()
    print(state.shape)
    

if __name__ == "__main__":
    app.run(test)
    