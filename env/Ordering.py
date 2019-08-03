import gym
import pandas as pd
import numpy as np

MAX_ORDER_SIZE = 40000
INITAL_AMOUNT = 200
MAX_WAREHOUSE_SPACE = 400000
FRAME_SIZE = 5


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
            shape=(FRAME_SIZE, self.number_of_products), dtype=np.int64
        )

        self.current_step = 0

        self.ware_amount = [0, 0]

    def _next_observation(self):
        frame = np.array([self.df.loc[self.current_step: self.current_step+FRAME_SIZE-1, column] for column in self.df.columns[1:]])

        frame = frame.reshape((FRAME_SIZE, self.number_of_products))

        # frame = frame.item()

        # obs = np.append(frame, [self.ware_amount])
        # obs = self.df.loc[self.current_step]

        print(self.df.loc[self.current_step])

        self.ware_amount -= self.df.loc[self.current_step, self.df.columns[1:]]

        # return np.array(self.ware_amount, obs)
        return frame

    def _take_action(self, action):

        ware = int(action[0]/MAX_ORDER_SIZE)
        amount_to_order = action[1]

        self.ware_amount[ware] += amount_to_order
        # amount = action[1]

        # if action_type == 0:
        #     Boy good 1
            # self.ware_amount += amount

    def step(self, action):

        print(action)

        self._take_action(action)

        self.current_step += 1

        reward = 100 - sum(self.ware_amount)

        print("reward", reward)

        done = sum(self.ware_amount) <= 0

        print("done", done)

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):

        self.ware_amount = INITAL_AMOUNT
        self.current_step = 0

        return self._next_observation()


if __name__ == "__main__":
    print(np.array([0]*2))
    wh = WareHouse(pd.read_csv("../data/orders2.csv"))

    print("action space", wh.action_space)
    print("observation space", wh.observation_space)

    print("step", wh.step([0,0]))

    print("next observation", wh._next_observation())
