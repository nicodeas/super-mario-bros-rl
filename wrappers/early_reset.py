import gym


class EarlyResetWrapper(gym.Wrapper):
    # NOTE: This wrapper was created for experimental purposes
    # this was used so that we could end the episode on a certain x position threshold
    # An early reset used to 'kill' mario when he reaches a certain x position in the level
    def __init__(self, env, x_pos_threshold: int):
        super().__init__(env)
        self.x_pos_threshold = x_pos_threshold

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info["x_pos"] > self.x_pos_threshold:
            truncated = True
            terminated = True
        return obs, reward, terminated, truncated, info
