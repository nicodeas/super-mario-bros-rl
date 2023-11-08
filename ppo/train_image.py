from .image_agent import Agent
import gym
from nes_py.wrappers import JoypadSpace
from wrappers.step_wrapper import CustomStepWrapper
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from tqdm.rich import tqdm
from gym.wrappers import (
    GrayScaleObservation,
    FrameStack,
    ResizeObservation,
    RecordEpisodeStatistics,
)
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
)


def make_env(gym_id, render=False):
    def func():
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

    return func


if __name__ == "__main__":
    gym_id = "SuperMarioBros-v0"
    lr = 1e-4
    total_steps = 1_000_000
    num_envs = 8
    num_steps = 512
    lambda_ = 0.95  # gae lambda coefficient
    gamma = 0.90  # discount coefficient
    epsilon = 0.2  # the clipping coefficient
    num_minibatches = 4
    update_epochs = 10
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    device = "cuda"

    writer = SummaryWriter(f"runs/{gym_id}_{total_steps}")

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches

    envs = gym.vector.SyncVectorEnv([make_env(gym_id) for _ in range(num_envs)])
    agent = Agent(envs).to(device)

    # Paper originally uses SGD but suggests ADAM for better performance
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    observations = torch.zeros(
        (num_steps, num_envs) + envs.single_observation_space.shape
    ).to(device)

    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(
        device
    )
    log_probabilities = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)

    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_steps // batch_size

    for update in tqdm(range(1, num_updates + 1)):
        # anneal learning rate
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * lr
        optimizer.param_groups[0]["lr"] = lrnow

        # POLICY ROLLOUT
        # Run policy pi_old in environment for T timesteps
        # Then compute advantage estimates for each time step
        for step in range(0, num_steps):
            global_step += 1 * num_envs
            observations[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_critic(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            log_probabilities[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

            # Visualisation
            for item in info.get("final_info", []):
                if item is not None:
                    writer.add_scalar(
                        "charts/episodic_return", item["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", item["episode"]["l"], global_step
                    )
                    # only log the first to reduce noise
                    break

        # GENERALIZED ADVANTAGE ESTIMATIONS
        with torch.no_grad():
            # inspired by https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/runner.py#L65
            # calculate GAE
            next_value = agent.critic_action(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    # nextnonterminal will be 0.0 if next state is a terminal state
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                # delta_t = r_t + lambda * V(s_{t+1})- V(s_t)
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]

                # A_t = delta_t + (gamma*lambda) * delta_{t+1} + ... + (gamma * lambda)^{T-t+1} * delta_{T-1}
                advantages[t] = lastgaelam = (
                    delta + gamma * lambda_ * nextnonterminal * lastgaelam
                )

            returns = advantages + values

        # flatten the batch
        batch_observations = observations.reshape(
            (-1,) + envs.single_observation_space.shape
        )

        batch_log_probabilities = log_probabilities.reshape(-1)
        batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        # Optimizing the policy and value network
        batch_inds = np.arange(batch_size)
        clip_fractions = []
        for epoch in range(update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                minibatch_inds = batch_inds[start:end]

                # forward pass
                # provide action to function so agent does not sample new actions
                _, new_log_prob, entropy, newvalue = agent.get_action_and_critic(
                    batch_observations[minibatch_inds],
                    batch_actions.long()[minibatch_inds],
                )

                log_ratio = new_log_prob - batch_log_probabilities[minibatch_inds]

                # r_t(theta) in original paper
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # kl helps understand how aggressively the policy updates
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fractions += [
                        ((ratio - 1.0).abs() > epsilon).float().mean().item()
                    ]

                minibatch_advantages = batch_advantages[minibatch_inds]

                # normalise advantages
                # add 1e-8 to prevent zero division error
                minibatch_advantages = (
                    minibatch_advantages - minibatch_advantages.mean()
                ) / (minibatch_advantages.std() + 1e-8)

                # Policy loss clipping
                policy_gradient_loss1 = -minibatch_advantages * ratio
                policy_gradient_loss2 = -minibatch_advantages * torch.clamp(
                    ratio, 1 - epsilon, 1 + epsilon
                )
                policy_gradient_loss = torch.max(
                    policy_gradient_loss1, policy_gradient_loss2
                ).mean()

                # Value loss clipping
                newvalue = newvalue.view(-1)

                # clip value loss
                value_loss_unclipped = (newvalue - batch_returns[minibatch_inds]) ** 2
                value_clipped = batch_values[minibatch_inds] + torch.clamp(
                    newvalue - batch_values[minibatch_inds],
                    -epsilon,
                    epsilon,
                )

                value_loss_clipped = (
                    value_clipped - batch_returns[minibatch_inds]
                ) ** 2
                value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                value_loss = 0.5 * value_loss_max.mean()

                # entropy is a measure of chaos in an action probability distribution
                # maximising entropy encourages more exploration
                # The idea is to minimise value loss and policy loss but maximise entropy loss
                entropy_loss = entropy.mean()
                # add entropy bonus to ensure sufficient exploration
                loss = (
                    policy_gradient_loss
                    - ent_coef * entropy_loss
                    + value_loss * vf_coef
                )

                # optimise and back propagate
                optimizer.zero_grad()
                loss.backward()

                # prevent exploding gradient that leads to aggressive policy update
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        predicted, actual = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
        variance_actual = np.var(actual)
        explained_var = (
            np.nan
            if variance_actual == 0
            else 1 - np.var(actual - predicted) / variance_actual
        )

    envs.close()
    writer.close()

    # evaluation code
    env = make_env(gym_id=gym_id, render=True)()
    done = True
    with torch.no_grad():
        for i in range(100000):
            if done:
                next_obs, info = env.reset()
                done = False
            else:
                action, logprob, _, value = agent.get_action_and_critic(
                    torch.unsqueeze(torch.asarray(next_obs), 0)
                )
                next_obs, reward, terminated, truncated, info = env.step(
                    action.cpu().numpy().item()
                )
                done = terminated or truncated
