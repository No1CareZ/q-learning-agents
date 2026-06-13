"""Just main."""

from agents import (
    RationalQLearningAgent,
    ProspectTheoryQLearningAgent,
    RiskSensitiveQLearningAgent,
)

from environment import Environment


def train_agent(agent, env, num_episodes=15000):
    """Procedure for training agents."""
    for episode in range(num_episodes):
        if episode % 1000 == 0:
            print(episode)
        state = env.reset()
        done = False  # Finished? (Or failed miserably)
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        agent.update_epsilon()


def test_agent(agent, env, num_episodes=300):
    """Procedure for testing agents."""
    successes = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        if reward == env.success_reward:
            successes += 1
    return successes / num_episodes


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

    print("All ready let's start")
    i = 0
    for name, agent in agents.items():
        print(f"Training {name}")
        train_agent(agent, env)
        i += 1

    print("Saving all")
    for name, agent in agents.items():
        agent.save()

    print("All trained let's test")
    i = 0
    for name, agent in agents.items():
        print(f"Testing {name}")
        print(test_agent(agent, env))
        i += 1
