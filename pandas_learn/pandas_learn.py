import pandas as pd
import numpy as np
from pandas import DataFrame

# data = DataFrame(np.arange(16).reshape(4, 4),
#                 index = ['Ohio', 'Colorado', 'Utah', 'New York'],
#                 columns = ['one', 'two', 'three', 'four'])

# print(data)

# transform = lambda x: x[:4].upper()

# data.index = data.index.map(transform)
# print(data)

data = pd.DataFrame(np.random.randn(1000, 4))
print(data)
print(data.describe())

print(np.abs(data) > 3)

print((data[np.abs(data) > 3]))