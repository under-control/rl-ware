import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.Ordering import WareHouse

import pandas as pd



df = pd.read_csv("data/orders.csv")

print(df.head())
print(len(df.columns))

