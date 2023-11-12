import gym
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from gym.wrappers import (
    GrayScaleObservation,
    FrameStack,
    ResizeObservation,
    RecordEpisodeStatistics,
)
from wrappers.step_wrapper import CustomStepWrapper


def make_env(gym_id, render=False):
    def thunk():
        if render:
            env = gym.make(gym_id, apply_api_compatibility=True, render_mode="human")
        else:
            env = gym.make(
                gym_id, apply_api_compatibility=True, render_mode="rgb_array"
            )
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = CustomStepWrapper(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env)
        env = FrameStack(env, 4)
        env = RecordEpisodeStatistics(env)
        return env

    return thunk


if __name__ == "__main__":
    # change to 1 for no multiprocessing
    env_id = "SuperMarioBros-7-3-v0"  # change to SuperMarioBros-3-4-v0 for other model provided
    model_location = ""

    model = PPO.load(model_location)
    env = DummyVecEnv([make_env(env_id, render=True)])
    state = env.reset()
    while True:
        action, _ = model.predict(state)
        state, rewards, done, info = env.step(action)
