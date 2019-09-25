import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.Ordering import WareHouse

import pandas as pd
import tensorflow as tf

df = pd.read_csv("data/orders2n.csv")

n_cpu = 12
envs = DummyVecEnv([lambda: WareHouse(df) for i in range(n_cpu)])

model = PPO2(MlpPolicy, envs, verbose=1, tensorboard_log="/home/king/Desktop/exp/warehouse/")
model.is_tb_set = False


def callback(locals_, globals_):
    self_ = locals_['self']
    produkt1 = self_.env.envs[00].ware_amount[0]
    produkt2 = self_.env.envs[00].ware_amount[1]
    summary1 = tf.Summary(value=[tf.Summary.Value(tag='produkt1', simple_value=produkt1)])
    summary2 = tf.Summary(value=[tf.Summary.Value(tag='produkt2', simple_value=produkt2)])
    locals_['writer'].add_summary(summary1, self_.num_timesteps)
    locals_['writer'].add_summary(summary2, self_.num_timesteps)

    produkt1sr10 = self_.env.envs[10].ware_amount[0]
    produkt2sr10 = self_.env.envs[10].ware_amount[1]
    summary1 = tf.Summary(value=[tf.Summary.Value(tag='produkt1sr10', simple_value=produkt1sr10)])
    summary2 = tf.Summary(value=[tf.Summary.Value(tag='produkt2sr10', simple_value=produkt2sr10)])
    locals_['writer'].add_summary(summary1, self_.num_timesteps)
    locals_['writer'].add_summary(summary2, self_.num_timesteps)

    return True


model.learn(total_timesteps=20000000, callback=callback)

obs = envs.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = envs.step(action)
    envs.render()
