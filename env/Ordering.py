import gym
import pandas as pd
import numpy as np

MAX_ORDER_SIZE = 40000
INITAL_AMOUNT = 200
MAX_WAREHOUSE_SPACE = 400000
FRAME_SIZE = 7


class WareHouse(gym.Env):
    """ Warehouse management enviroment based on Open AI gym environments """

    def __init__(self, df):
        super(WareHouse, self).__init__()

        self.df = df

        self.number_of_products = len(df.columns) - 1

        self.action_space = gym.spaces.Box(
            low=(np.array([0]*self.number_of_products)), high=(np.array([MAX_ORDER_SIZE]*self.number_of_products))
        )

        self.observation_space = gym.spaces.Box(
            # low=(np.array([0]*self.number_of_products)), high=np.array([MAX_WAREHOUSE_SPACE]*self.number_of_products),
            low=0, high=MAX_WAREHOUSE_SPACE,
            shape=(self.number_of_products+1,), dtype=np.int64
        )

        self.current_step = 0

        self.ware_amount = 0

    def _next_observation(self):
        frame = np.array([self.df.loc[self.current_step: self.current_step+FRAME_SIZE-1, column] for column in self.df.columns[1:]])

        # obs = np.append(frame)
        obs = self.df.loc[self.current_step]

        self.ware_amount -= self.df.loc[self.current_step]


        return obs

    def _take_action(self, action):

        action_type = action[0]
        amount = action[1]

        if action_type == 0:
            # Boy good 1
            self.ware_amount += amount

    def step(self, action):

        print(action)

        self._take_action(action)

        self.current_step += 1

        reward = 100 - self.ware_amount

        done = self.ware_amount <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):

        self.ware_amount = INITAL_AMOUNT

        return self._next_observation()



if __name__ == "__main__":
    print(np.array([0]*2))
    wh = WareHouse(pd.read_csv("../data/orders.csv"))

    print("action space", wh.action_space)
    print("obesrvation space",wh.observation_space)

    print(wh._next_observation())