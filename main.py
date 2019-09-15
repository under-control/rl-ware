import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.Ordering import WareHouse

import pandas as pd

df = pd.read_csv("data/orders2.csv")

n_cpu = 12
envs = DummyVecEnv([lambda: WareHouse(df) for i in range(n_cpu)])

model = PPO2(MlpPolicy, envs, verbose=1, tensorboard_log="/home/king/Desktop/exp/warehouse/")
model.learn(total_timesteps=100000)

env = envs[0]
obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
