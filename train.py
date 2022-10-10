import numpy as np
import sys
sys.path.append(r'/mnt/Edisk/andrew/Self_learning_MC')
import packing

from stable_baselines3 import PPO
from environment import ASC

# create environment
env = ASC(packing, np.sqrt(93./64.), 24)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="/mnt/Edisk/andrew/Self_learning_MC/tensorboard/300_tensorboard-v1/")
# model.learn(total_timesteps=int(1e3), tb_log_name="new_penalty_run")
for i in range(int(1e3)):
    model.learn(total_timesteps=int(1e5), tb_log_name="first_run", reset_num_timesteps=False)
    model.save("asc-v1","asc-v1")


obs = env.reset()
for i in range(int(1e2)):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()


# tensorboard --logdir=tensorboard/300_tensorboard-v1/first_run_0 --port 8123