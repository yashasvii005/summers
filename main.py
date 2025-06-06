
import numpy as np
import pandas as pd

df = pd.read_csv("placement.csv")
print(df.head(2))

x = df.drop(columns=['placement'])
y = df['placement']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=42)

print(x_train.head(3))