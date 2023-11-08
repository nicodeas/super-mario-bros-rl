import gym
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from gym.wrappers import (
    GrayScaleObservation,
    FrameStack,
    ResizeObservation,
    RecordEpisodeStatistics,
)
from wrappers.step_wrapper import CustomStepWrapper
from wrappers.early_reset import EarlyResetWrapper


def make_env(gym_id, render=False):
    def thunk():
        if render:
            env = gym.make(gym_id, apply_api_compatibility=True, render_mode="human")
        else:
            env = gym.make(
                gym_id, apply_api_compatibility=True, render_mode="rgb_array"
            )
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = EarlyResetWrapper(env, 1150)
        env = CustomStepWrapper(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env)
        env = FrameStack(env, 4)
        env = RecordEpisodeStatistics(env)
        return env

    return thunk


if __name__ == "__main__":
    env_id = "SuperMarioBros-1-1-v0"
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path="truncated_image_models",
        name_prefix="mario_image",
    )

    env = DummyVecEnv([make_env(env_id)])

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="truncated_runs",
        learning_rate=1e-4,
        gamma=0.9,
        ent_coef=0.01,
        batch_size=16,
        n_epochs=10,
        n_steps=512,
    )
    model.learn(100_000, progress_bar=True, callback=checkpoint_callback)
    model.save("truncated_image_mario")
