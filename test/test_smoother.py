import pandas as pd

from src.behavior.smoother import TrajectorySmoother

df = pd.read_csv("datasets/trajectory/normal.csv")

track = df[df["track_id"] == 1]

smoother = TrajectorySmoother(alpha=0.3)

result = smoother.smooth_track(track)

print(result[[
    "frame",
    "x",
    "smooth_x",
    "y",
    "smooth_y"
]].head(20))