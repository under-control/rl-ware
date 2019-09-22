import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_of_products = 2

df = pd.DataFrame(np.random.randint(0, 100, size=(5000, num_of_products)), columns=["Produkt 1", "Produkt 2"])

print(df.head())

df.hist(bins=20, figsize=(9, 4), edgecolor='black', linewidth=1.5)
plt.show()

df.to_csv("../data/orders" + str(num_of_products) + ".csv")
