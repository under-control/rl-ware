import gym

MAX_ORDER_SIZE = 40000
MAX_WAREHOUSE_SPACE = 400000


class WareHouse(gym.Env):

    def __init__(self, df):
        super(WareHouse, self).__init__()

        self.df = df

        self.number_of_products = len(df.columns-1)

        self.action_space = gym.spaces.Box(
            low=[0]*self.number_of_products, high=[MAX_ORDER_SIZE]*self.number_of_products)

        self.observation_space = gym.spaces.Box(
            low=[0]*self.number_of_products, high=MAX_WAREHOUSE_SPACE, shape=(len(df.columns), 10))

