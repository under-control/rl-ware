import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randint(0,100, size=(5000, 2)), columns=["Produkt 1", "Produkt 2"])

print(df.head())

df.hist(bins=100)
plt.show()

print(len(df))

df.to_csv("../data/orders2.csv")

