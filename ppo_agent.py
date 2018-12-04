from collections import namedtuple

import numpy as np
import torch
from torch import optim

from model import ActorCriticModel
from utils import to_tensor, device


class PPOAgent:
    def __init__(self, env,
                 lr=3e-4,
                 gamma=0.995,
                 gae_lambda=0.95,
                 num_epochs=10,
                 mini_batch_size=512,
                 clip_eps=0.2,
                 coeff_entropy_loss=0.01,
                 coeff_value_loss=0.5
                 ) -> None:
        super().__init__()

        self._lr = lr
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._num_epochs = num_epochs
        self._mini_batch_size = mini_batch_size
        self._clip_eps = clip_eps
        self._coeff_entropy_loss = coeff_entropy_loss
        self._coeff_value_loss = coeff_value_loss

        self.env = env
        self.model = ActorCriticModel(env.num_states, env.num_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self._lr)

    def run_episode(self, max_steps):
        trajectories, episode_rewards, steps = self._collect_trajectories(max_steps)
        states, actions, returns, log_probs_old, advantages = self._compute_advantage_estimates(trajectories)
        self._learn(states, actions, returns, log_probs_old, advantages)

        return episode_rewards, steps

    def _collect_trajectories(self, max_steps):
        num_agents = self.env.num_agents

        episode_rewards = np.zeros((num_agents,))
        trajectories = []
        for a in range(num_agents):
            trajectories.append([])

        states = self.env.reset(train_mode=True)
        i_step = 0
        while True:
            i_step += 1
            if i_step >= max_steps + 1:  # Last step is used only for processing
                break
            dist, values = self.model(to_tensor(states))

            actions = dist.sample()
            log_probs = self.model.log_prob(dist, actions)

            actions = actions.detach().numpy()

            next_states, rewards, dones, _ = self.env.step(actions)

            values = values.detach().numpy()
            log_probs = log_probs.detach().numpy()

            for a in range(num_agents):
                episode_rewards[a] += rewards[a]

                e = Experience(states[a], actions[a], rewards[a], dones[a], values[a], log_probs[a])
                trajectories[a].append(e)

            states = next_states

        return trajectories, episode_rewards, i_step - 1

    def _compute_advantage_estimates(self, trajectories):
        states = []
        actions = []
        returns = []
        log_probs = []
        advantages = []

        for traj in trajectories:
            advantage = 0
            return_ = traj[-1].value

            for i in reversed(range(len(traj) - 1)):
                exp = traj[i]
                reward = np.asarray([exp.reward])
                done = np.asarray([exp.done])
                return_ = reward + self._gamma * (1 - done) * return_
                next_value = traj[i + 1].value

                # Generalized Advantage Estimation
                # δᵗ = rᵗ + γ * V(sᵗ⁺ⁱ) − V(sᵗ)
                # Â(∞)= ∑ γˡ * δᵗ
                delta = reward + self._gamma * (1 - done) * next_value - exp.value
                advantage = advantage * (self._gae_lambda * self._gamma) * (1 - done) + delta

                states.append(exp.state)
                actions.append(exp.action)
                returns.append(return_)
                log_probs.append(exp.log_prob)
                advantages.append(advantage)

        # Required only for debugging
        states = list(reversed(states))
        actions = list(reversed(actions))
        returns = list(reversed(returns))
        log_probs = list(reversed(log_probs))
        advantages = list(reversed(advantages))

        states = to_tensor(states)
        actions = to_tensor(actions)
        returns = to_tensor(returns)
        log_probs = to_tensor(log_probs)
        advantages = to_tensor(advantages)

        # Normalize the advantages
        advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
    
        return states, actions, returns, log_probs, advantages

    def _learn(self, states, actions, returns, log_probs_old, advantages):
        num_experiences = len(states)

        for i_epoch in range(self._num_epochs):
            indices = np.asarray(list(range(num_experiences)))
            np.random.shuffle(indices)

            remaining = num_experiences % self._mini_batch_size
            batches = indices[:num_experiences - remaining].reshape(-1, self._mini_batch_size).tolist()
            if remaining > 0:
                batches.append(indices[-remaining:])

            for idxs in batches:
                dist, values = self.model(states[idxs])
                log_probs = self.model.log_prob(dist, actions[idxs])

                # rᵗ(θ) = πᶿ(aᵗ|sᵗ) / π_oldᶿ(aᵗ|sᵗ)
                ratio = torch.exp(log_probs - log_probs_old[idxs])

                # Lᶜˡⁱᵖ = Ê[min(
                #       rᵗ(θ) * Â,
                #       clip(rᵗ(θ), 1 - ε, 1 + ε) * Âᵗ
                # )]
                clip_loss = torch.min(
                    ratio * advantages[idxs],
                    torch.clamp(ratio, 1. - self._clip_eps, 1. + self._clip_eps) * advantages[idxs]
                ).mean(0)

                # Lᵛᶠ
                value_loss = torch.mean(torch.pow(returns[idxs] - values, 2))

                # S[πᶿ](sᵗ)
                entropy_loss = torch.mean(dist.entropy().unsqueeze(-1))

                # Loss = Lᶜˡⁱᵖ - c1 * Lᵛᶠ + c2 * S[πᶿ](sᵗ)
                loss = clip_loss - self._coeff_value_loss * value_loss + self._coeff_entropy_loss * entropy_loss

                self.optimizer.zero_grad()
                (-loss).backward() # Maximize the objective
                self.optimizer.step()


Experience = namedtuple("Experience", ("state", "action", "reward", "done", "value", "log_prob"))
