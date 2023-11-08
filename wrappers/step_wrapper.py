import gym


class CustomStepWrapper(gym.Wrapper):
    def __init__(self, env):
        # ends the episode when a life is used up
        # This helps monitor the true episodics return of the run
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.score = 0

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # get true episodic return per run
        lives = info["life"]
        if 0 < lives < self.lives:
            terminated = True
            return obs, reward, terminated, truncated, info

        self.lives = lives
        reward += (info["score"] - self.score) / 40
        self.score = info["score"]
        return obs, reward / 10.0, terminated, truncated, info

    def reset(self, **_):
        self.score = 0
        if self.was_real_done:
            obs, info = self.env.reset()
        else:
            # take a no-op step to get the states going again
            obs, _, _, _, info = self.env.step(0)
            self.lives = info["life"]

        return obs, info
