import pandas as pd
import matplotlib.pyplot as plt

# task 1
data = pd.read_csv("datasets/wine-clustering.csv")
print(data)

# task 1.2
plt.scatter(data['Alcohol'].to_numpy(), data['Proline'].to_numpy())
plt.show()