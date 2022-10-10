import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(r'/mnt/Edisk/andrew/Self_learning_MC')
import packing

from stable_baselines3 import PPO
from environment import ASC

env = ASC(packing, np.sqrt(93./64.), 24)
model = PPO.load("/mnt/Edisk/andrew/Self_learning_MC/asc-v21.zip")

info_list = []
obs = env.reset()
for i in tqdm(range(int(1e5))):
# for i in tqdm(range(int(1e5))):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    info_list.append(info)
    if done:
        env.reset()
 
pd.DataFrame(info_list).to_csv("/mnt/Edisk/andrew/Self_learning_MC/outcomes/analysis-v1.csv")