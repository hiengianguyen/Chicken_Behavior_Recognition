import pandas as pd

df = pd.read_csv("datasets/behavior/normal.csv")

print(df[["speed", "standing_time"]].head(30))
