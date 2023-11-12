import gym
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from gym.wrappers import (
    GrayScaleObservation,
    FrameStack,
    ResizeObservation,
    RecordEpisodeStatistics,
)
from wrappers.step_wrapper import CustomStepWrapper
import multiprocessing as mp


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
    num_processes = mp.cpu_count()  # 8 used in training
    env_id = "SuperMarioBros-7-3-v0"
    device = "cuda"  # use mps for mac or cpu if neither are available
    num_training_steps = 2_000_000

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // num_processes, 1),
        save_path="./image_models_7_3",
        name_prefix="mario_image_7_3",
    )

    if num_processes == 1:
        env = DummyVecEnv([make_env(env_id)])
    else:
        env = SubprocVecEnv(
            [make_env(env_id) for _ in range(num_processes)],
            start_method="spawn",
        )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="runs",
        learning_rate=1e-4,
        gamma=0.9,
        ent_coef=0.01,
        batch_size=16,
        n_epochs=10,
        n_steps=512,
        device=device,
    )
    model.learn(num_training_steps, progress_bar=True, callback=checkpoint_callback)

    # uncomment to evaluate model
    # model = PPO.load("<path to model>")  # to load custom model
    # env = DummyVecEnv([make_env(env_id, render=True)])
    # state = env.reset()
    # while True:
    #     action, _ = model.predict(state)
    #     state, rewards, done, info = env.step(action)
