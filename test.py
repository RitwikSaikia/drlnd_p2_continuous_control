import numpy as np

from env import UnityEnv, unity_filename
from model import ActorCriticModel
from utils import to_tensor


env_name = "Reacher"
# env_name = "Crawler"
env = UnityEnv(env_name)

model = ActorCriticModel(env.num_states, env.num_actions)
model.load_model("checkpoints/%s.ppo.best" % env.name)

num_episodes = 5

for i_episode in range(num_episodes):

    states = env.reset()
    while True:
        dist, values = model(to_tensor(states))

        actions = dist.sample()
        next_states, rewards, dones, _ = env.step(actions.cpu().numpy())

        states = next_states

        if np.any(dones):
            break
