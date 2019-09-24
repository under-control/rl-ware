import gym
import pandas as pd
import numpy as np

MAX_ORDER_SIZE = 400
INITAL_AMOUNT = 100
MAX_WAREHOUSE_SPACE = 4000
FRAME_SIZE = 5


class WareHouse(gym.Env):
    """ Warehouse management environment based on Open AI gym environments """

    def __init__(self, df):
        super(WareHouse, self).__init__()

        self.df = df

        self.number_of_products = len(df.columns) - 1

        self.action_space = gym.spaces.Box(
            low=(np.array([0]*self.number_of_products)), high=(np.array([MAX_ORDER_SIZE]*self.number_of_products))
        )

        self.observation_space = gym.spaces.Box(
            low=0, high=MAX_WAREHOUSE_SPACE,
            shape=(FRAME_SIZE, self.number_of_products), dtype=np.int64
        )

        self.current_step = 0

        self.episode_reward = 0

        self.ware_amount = [0] * self.number_of_products

    def _next_observation(self):
        frame = np.array([self.df.loc[self.current_step: self.current_step+FRAME_SIZE-1, column] for column in self.df.columns[1:]])

        frame = frame.reshape((FRAME_SIZE, self.number_of_products))

        self.ware_amount -= self.df.loc[self.current_step, self.df.columns[1:]]

        return frame

    def _take_action(self, action):

        for i in range(self.number_of_products):
            self.ware_amount[i] += action[i]

    def step(self, action):

        self._take_action(action)

        self.current_step += 1

        reward = 800

        for i in range(self.number_of_products):
            reward -= abs(self.ware_amount[i])

        lack = any(i < 0 for i in self.ware_amount)

        if lack:
            for i in self.ware_amount:
                if i < 0:
                    reward += 20 * i

        if any(i < -100 for i in self.ware_amount):
            reward -= 1000
            done = True
        else:
            done = False

        if self.current_step > 1000:
            done = True
            reward += 1000

        self.episode_reward += reward

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):

        for i in range(self.number_of_products):
            self.ware_amount[i] = INITAL_AMOUNT
        self.current_step = 0
        self.episode_reward = 0

        return self._next_observation()

    def render(self, mode='human'):

        print("Step", self.current_step)
        print("Episode reward", self.episode_reward)


if __name__ == "__main__":
    print(np.array([0]*2))
    wh = WareHouse(pd.read_csv("../data/orders2.csv"))

    print("action space", wh.action_space)
    print("observation space", wh.observation_space)

    print("step", wh.step([0,0]))

    print("next observation", wh._next_observation())

    print(wh.number_of_products)
