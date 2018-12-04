#!/usr/bin/env python3
import logging
import os
from collections import deque

from env import *
from plot import plot_scores
from ppo_agent import PPOAgent
from utils import *

logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger().setLevel(logging.DEBUG)


def train(agent, model_file_prefix,
          num_episodes=2000,
          max_steps=1000,
          solved_score=30,
          window_size=100):
    all_scores = []
    scores = deque(maxlen=window_size)
    best_score = -np.inf

    solved = False
    i_episode = 0
    stop_episode = num_episodes
    while True:
        i_episode += 1
        if i_episode >= stop_episode:
            break

        episode_rewards, steps = agent.run_episode(max_steps)

        score = np.mean(episode_rewards)
        scores.append(score)
        all_scores.append(score)
        avg_score = np.mean(scores)
        if len(scores) >= window_size and avg_score > best_score:
            best_score = avg_score
            agent.model.save_model("%s.best" % model_file_prefix)

        msg = "[%d/%d] score=%.2f, avg = %.2f, best = %.2f, steps = %d" % (
        i_episode, num_episodes, score, avg_score, best_score, steps)
        logger.info(msg)

        np.save("scores.npy", all_scores)

        if not solved and best_score > solved_score:
            logger.info("Environment solved in %d episodes" % (i_episode))
            agent.model.save_model("%s.solved" % model_file_prefix)
            stop_episode = i_episode + 150  # Run for few more episodes
            solved = True

    np.save("scores-final.npy", all_scores)
    plot_scores(scores, solved_score=solved_score)


def main(env_name, seed=None):
    if seed is not None:
        set_seed(seed)

    env = UnityEnv(env_name, seed=seed, no_graphics=True)

    model_file_prefix = "checkpoints/%s.ppo" % env.name

    agent = PPOAgent(env)

    train(agent, model_file_prefix)


if __name__ == '__main__':
    env_name = "Reacher"
    # env_name = "Crawler"
    main(env_name, seed=0)
