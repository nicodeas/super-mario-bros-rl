import gym
from gym.spaces import Box
import numpy as np
from imaging import locate_objects

BLOCK_SIZE = 16


def get_state(obs, info):
    mario_status = info["status"]
    object_locations = locate_objects(obs, mario_status)

    if not object_locations.get("mario"):
        return np.array([0] * 14, dtype=np.float32)

    mario = object_locations["mario"][0]
    mario_x = mario.x_min
    mario_y = mario.y_max

    enemy_locations = object_locations["enemy"]
    block_locations = object_locations["block"]
    pipe_locations = []
    question_blocks = []

    for block in block_locations:
        if block.bbox_type == "question_block":
            question_blocks.append(block)
        elif block.bbox_type == "pipe":
            pipe_locations.append(block)

    first_enemy = False
    enemy_x = 24 * BLOCK_SIZE
    enemy_y = 24 * BLOCK_SIZE

    second_enemy = False
    enemy2_x = 24 * BLOCK_SIZE
    enemy2_y = 24 * BLOCK_SIZE

    for enemy_location in sorted(enemy_locations, key=lambda x: x.x_min):
        x_distance = enemy_location.x_max - mario.x_min
        y_distance = abs(mario.y_max - enemy_location.y_min)
        if x_distance >= 0:
            if not first_enemy:
                enemy_x = x_distance
                enemy_y = y_distance
                first_enemy = True
            elif not second_enemy:
                enemy2_x = x_distance
                enemy2_y = y_distance
                second_enemy = True
                break

    enemy_behind_x = 24 * BLOCK_SIZE
    enemy_behind_y = 24 * BLOCK_SIZE
    for enemy_location in sorted(enemy_locations, key=lambda x: x.x_min, reverse=True):
        x_distance = mario.x_min - enemy_location.x_max
        y_distance = abs(mario.y_max - enemy_location.y_min)
        if x_distance > 0:
            enemy_behind_x = x_distance
            enemy_behind_y = y_distance
            break

    pipe_x = 24 * BLOCK_SIZE
    pipe_y = 24 * BLOCK_SIZE
    for pipe in pipe_locations:
        x_distance = pipe.x_min - mario.cx
        if x_distance > 0:
            pipe_x = pipe.x_min - mario.x_max
            pipe_y = pipe.y_min - mario.y_max
            break

    hole_x = 24 * BLOCK_SIZE  # distance to hole
    hole_width = 0
    floor = [
        block
        for block in block_locations
        if block.y_min == 208 and block.x_max >= mario.x_min
    ]
    floors_sorted = sorted(floor, key=lambda x: x.x_min)
    for i in range(1, len(floors_sorted)):
        if (
            floors_sorted[i].x_min - floors_sorted[i - 1].x_max >= 2 * 32
        ):  # NOTE: more than 32 it sometimes detects goombas, problem with imaging code
            hole_x = floors_sorted[i - 1].x_max - mario.x_max
            hole_width = floors_sorted[i].x_min - floors_sorted[i - 1].x_max
            break

    question_x = 24 * BLOCK_SIZE
    question_y = 24 * BLOCK_SIZE
    for question in question_blocks:
        x_distance = question.x_min - mario.cx
        if x_distance > 0:
            question_x = question.x_min - mario.x_max
            question_y = question.y_min - mario.y_max
            break

    return np.array(
        [
            mario_x,
            mario_y,
            enemy_x,
            enemy_y,
            enemy2_x,
            enemy2_y,
            enemy_behind_x,
            enemy_behind_y,
            pipe_x,
            pipe_y,
            hole_x,
            hole_width,
            question_x,
            question_y,
        ],
        dtype=np.float32,
    )


class SimpleMarioObservation(gym.Wrapper):
    def __init__(self, env):
        """skip_frames: num of frames to skip, also applies the same action for a particular frame"""
        super(SimpleMarioObservation, self).__init__(env)
        self.state_space_size = 14
        self.observation_space = Box(
            low=-16 * 24, high=16 * 24, shape=(self.state_space_size,), dtype=np.float32
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # obs is going to be our state space after imaging code at the end
        obs = (
            get_state(obs, info)
            if not done
            else np.array([0] * self.state_space_size, dtype=np.float32)
        )

        return obs, reward, terminated, truncated, info

    def reset(self, **_):
        self.env.reset()
        return np.array([0] * self.state_space_size, dtype=np.float32), {}
