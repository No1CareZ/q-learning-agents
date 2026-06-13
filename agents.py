"""Agents."""

import numpy as np
import random
import os
import json

random.seed(67)


class BaseAgent:
    """."""

    def __init__(
        self,
        env,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.997,
        lr=0.1,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr

        # Table state_size by action_size
        self.state_size = env.get_state_size()
        self.action_size = env.get_action_size()

        # Table (dont forget that I will inherit it)
        self.table = None

    def get_action(self, state, training=True) -> int:
        """Acctually will return index of action."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self._get_q_values(state))

    def _get_q_values(self, state):
        """."""

    def update(self, state, action, reward, next_state, done):
        """."""

    def update_epsilon(self):
        """Epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _state_to_index(self, state):
        """."""
        delta_x_bins = 14 if hasattr(self.env, "delta_x_bins") else 15
        delta_y_bins = 14 if hasattr(self.env, "delta_y_bins") else 15
        battery_bins = 12 if hasattr(self.env, "battery_bins") else 10
        num_speeds = len(self.env.speed_values)

        return np.ravel_multi_index(
            state, (delta_x_bins, delta_y_bins, battery_bins, num_speeds)
        )

    def save(self, dir_path="saved_agents"):
        """Save."""
        os.makedirs(dir_path, exist_ok=True)

        np.save(f"{dir_path}/{self.__class__.__name__}_table.npy", self.table)

        params = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "lr": self.lr,
        }
        if hasattr(self, "alpha_p"):
            params.update(
                {
                    "alpha_p": self.alpha_p,
                    "beta_p": self.beta_p,
                    "lambda_p": self.lambda_p,
                }
            )
        if hasattr(self, "eta"):
            params["eta"] = self.eta

        with open(f"{dir_path}/{self.__class__.__name__}_params.json", "w") as f:
            json.dump(params, f)

    def load(self, dir_path="saved_agents"):
        """Load."""
        table = np.load(f"{dir_path}/{self.__class__.__name__}_table.npy")
        self.table = table

        with open(f"{dir_path}/{self.__class__.__name__}_params.json", "r") as f:
            params = json.load(f)

        for key, value in params.items():
            setattr(self, key, value)


class RationalQLearningAgent(BaseAgent):
    """."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # Fill table with 0
        self.table = np.zeros((self.state_size, self.action_size))

    def _get_q_values(self, state):
        """."""
        state_idx = self._state_to_index(state)
        return self.table[state_idx]

    def update(self, state, action, reward, next_state, done):
        """Update."""
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)

        best_next_q = np.max(self.table[next_state_idx])
        td_target = reward + (0 if done else self.gamma * best_next_q)

        td_error = td_target - self.table[state_idx][action]
        self.table[state_idx][action] += self.lr * td_error


class ProspectTheoryQLearningAgent(BaseAgent):
    """."""

    def __init__(self, env, alpha_p=0.88, beta_p=0.88, lambda_p=2.35, **kwargs):
        super().__init__(env, **kwargs)
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.lambda_p = lambda_p
        self.table = np.zeros((self.state_size, self.action_size))

    def _get_q_values(self, state):
        """."""
        state_idx = self._state_to_index(state)
        return self.table[state_idx]

    def _value_function(self, reward):
        """Update reward accordingly."""
        if reward >= 0:
            return reward**self.alpha_p
        else:
            return -self.lambda_p * ((-reward) ** self.beta_p)

    def update(self, state, action, reward, next_state, done):
        """Update."""
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)

        best_next_q = np.max(self.table[next_state_idx])
        td_target = reward + (0 if done else self.gamma * best_next_q)

        td_error = td_target - self.table[state_idx][action]
        self.table[state_idx][action] += self.lr * td_error


class RiskSensitiveQLearningAgent(BaseAgent):
    """."""

    def __init__(self, env, eta=1.0, **kwargs):
        super().__init__(env, **kwargs)
        self.eta = eta
        self.table = np.ones((self.state_size, self.action_size))
        self.failure_reward = -env.failure_reward

    def _get_q_values(self, state):
        """."""
        state_idx = self._state_to_index(state)
        return -np.log(self.table[state_idx]) / self.eta

    def update(self, state, action, reward, next_state, done):
        """Update."""
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)

        reward /= self.failure_reward  # let's try to normalize it

        if done:
            u_target = np.exp(-self.eta * reward)
        else:
            min_next_u = np.min(self.table[next_state_idx])
            u_target = np.exp(-self.eta * reward) * (min_next_u**self.gamma)

        self.table[state_idx][action] = (1 - self.lr) * self.table[state_idx][
            action
        ] + self.lr * u_target
