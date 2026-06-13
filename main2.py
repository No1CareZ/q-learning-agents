from agents import (
    RationalQLearningAgent,
    ProspectTheoryQLearningAgent,
    RiskSensitiveQLearningAgent,
)
from environment import Environment

import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
from tqdm import tqdm


def plot_learning_curves(train_histories, window_size=150):
    """Plot learning curves."""
    plt.figure(figsize=(12, 6))
    for name, history in train_histories.items():
        rewards = history["rewards"]
        if len(rewards) >= window_size:
            smoothed = np.convolve(
                rewards, np.ones(window_size) / window_size, mode="valid"
            )
            plt.plot(smoothed, label=name.replace("_", " ").title())
        else:
            plt.plot(rewards, label=name.replace("_", " ").title(), alpha=0.7)
    plt.xlabel("Эпизод")
    plt.ylabel("Скользящее среднее награды (окно=150)")
    plt.title("Кривые обучения")
    plt.legend()
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/learning_curves.png", dpi=300)
    plt.close()
    print("Saved chart: figures/learning_curves.png")


def plot_cdf_success_times(test_results):
    """Plot cdf."""
    plt.figure(figsize=(10, 6))
    for name, result in test_results.items():
        times = np.array(result["success_times"])
        if len(times) > 0:
            sorted_times = np.sort(times)
            cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
            plt.plot(sorted_times, cdf, label=f"{name} ({len(times)} успехов)")
    plt.xlabel("Шаги до успеха")
    plt.ylabel("CDF")
    plt.title("Функция распределения времени до успеха")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/cdf_success_times.png", dpi=300)
    plt.close()
    print("Saved chart: figures/cdf_success_times.png")


def plot_final_charge_distribution(test_results):
    """Plot distribution."""
    plt.figure(figsize=(12, 6))
    for i, (name, result) in enumerate(test_results.items()):
        plt.subplot(1, 3, i + 1)
        sns.histplot(result["final_charges"], bins=20, kde=True, color="blue")
        plt.title(name.replace("_", " ").title())
        plt.xlabel("Финальный заряд батареи")
        plt.ylabel("Частота")
    plt.suptitle("Распределение финального заряда батареи")
    plt.tight_layout()
    plt.savefig("figures/final_charge_distribution.png", dpi=300)
    plt.close()
    print("Saved chart: figures/final_charge_distribution.png")


def plot_trajectories(test_results, env, num_trajectories=4):
    """Visualize path."""
    plt.figure(figsize=(15, 10))
    colors = {"rational": "blue", "prospect": "green", "risk_sensitive": "red"}

    for i, (name, result) in enumerate(test_results.items()):
        plt.subplot(2, 2, i + 1)

        for traj in result["trajectories"][:num_trajectories]:
            x_coords = [p["position"][0] for p in traj]
            y_coords = [p["position"][1] for p in traj]
            plt.plot(
                x_coords,
                y_coords,
                alpha=0.7,
                color=colors[name],
                marker="o",
                markersize=2,
            )

        station = Circle(
            env.station_center,
            env.station_radius,
            color="green",
            alpha=0.3,
            label="Station",
        )
        plt.gca().add_patch(station)

        for obs in env.obstacles:
            rect = Rectangle(
                obs[0],
                obs[1][0] - obs[0][0],
                obs[1][1] - obs[0][1],
                color="red",
                alpha=0.3,
                label="Obstacle",
            )
            plt.gca().add_patch(rect)

        plt.xlim(env.x_min, env.x_max)
        plt.ylim(env.y_min, env.y_max)
        plt.title(f"{name.replace('_', ' ').title()}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().set_aspect("equal")
        plt.legend()

    plt.suptitle(f"Примеры траекторий (по {num_trajectories} на агента)")
    plt.tight_layout()
    plt.savefig("figures/trajectories.png", dpi=300)
    plt.close()
    print("Saved chart: figures/trajectories.png")


def train_agent(agent, env, num_episodes=15000):
    """Train agent."""
    rewards_history = []
    lengths_history = []
    collisions_history = []
    final_charges_history = []

    for episode in tqdm(
        range(num_episodes), desc=f"Training {agent.__class__.__name__}"
    ):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        collision_count = 0

        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            if info.get("collision", False):
                collision_count += 1
            step += 1

            agent.update(state, action, reward, next_state, done)
            state = next_state

        rewards_history.append(total_reward)
        lengths_history.append(step)
        collisions_history.append(collision_count)
        final_charges_history.append(info["battery"])

        agent.update_epsilon()

    return {
        "rewards": rewards_history,
        "lengths": lengths_history,
        "collisions": collisions_history,
        "final_charges": final_charges_history,
        "agent": agent,
    }


def test_agent(agent, env, num_episodes=300):
    """Test agent."""
    successes = 0
    success_times = []
    final_charges = []
    trajectories = []
    collision_counts = []

    for _ in tqdm(range(num_episodes), desc=f"Testing {agent.__class__.__name__}"):
        state = env.reset()
        done = False
        step = 0
        trajectory = []
        collision_count = 0

        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            trajectory.append(
                {
                    "position": info["position"],
                    "battery": info["battery"],
                    "speed": info["speed"],
                }
            )
            if info.get("collision", False):
                collision_count += 1
            step += 1
            state = next_state

        if reward == env.success_reward:
            successes += 1
            success_times.append(step)
        final_charges.append(info["battery"])
        trajectories.append(trajectory)
        collision_counts.append(collision_count)

    return {
        "success_rate": successes / num_episodes * 100,
        "avg_success_time": np.mean(success_times) if success_times else 0,
        "success_times": success_times,
        "final_charges": final_charges,
        "trajectories": trajectories,
        "collision_rate": (
            np.mean(collision_counts) / num_episodes * 100 if num_episodes > 0 else 0
        ),
    }


if __name__ == "__main__":
    env = Environment()

    agents = {
        "rational": RationalQLearningAgent(env, gamma=0.99, epsilon=1.0, lr=0.1),
        "prospect": ProspectTheoryQLearningAgent(
            env,
            gamma=0.99,
            epsilon=1.0,
            lr=0.1,
            alpha_p=0.88,
            beta_p=0.88,
            lambda_p=2.35,
        ),
        "risk_sensitive": RiskSensitiveQLearningAgent(
            env, gamma=0.99, epsilon=1.0, lr=0.1, eta=1.0
        ),
    }

    train_histories = {}
    for name, agent in agents.items():
        train_histories[name] = train_agent(agent, env)
        agents[name] = train_histories[name]["agent"]

    print("\nSaving...")
    for name, agent in agents.items():
        agent.save(f"saved_agents/{name}")
        print(f"  Saved {name}")

    print("\nTesting began...")
    test_results = {}
    for name, agent in agents.items():
        test_results[name] = test_agent(agent, env, num_episodes=300)

    print("\nPlotting...")
    plot_learning_curves(train_histories)
    plot_cdf_success_times(test_results)
    plot_final_charge_distribution(test_results)
    plot_trajectories(test_results, env)

    print("\nFinal metrics:")
    for name, result in test_results.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Success: {result['success_rate']:.2f}%")
        print(f"  Mean time till success: {result['avg_success_time']:.2f} steps")
        print(f"  Collision %: {result['collision_rate']:.2f}%")
