from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


class FeaturePlotter:

    def __init__(self, csv_path: str):

        self.csv_path = Path(csv_path)

        self.df = pd.read_csv(self.csv_path)

    def plot(self, feature_name: str):

        if feature_name not in self.df.columns:
            raise ValueError(f"{feature_name} not found.")

        plt.figure(figsize=(14,5))

        plt.plot(
            self.df["frame"],
            self.df[feature_name]
        )

        plt.title(feature_name)

        plt.xlabel("Frame")

        plt.ylabel(feature_name)

        plt.grid(True)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def compare(csv1, csv2, feature):

        df1 = pd.read_csv(csv1)

        df2 = pd.read_csv(csv2)

        plt.figure(figsize=(14,5))

        plt.plot(
            df1["frame"],
            df1[feature],
            label="Normal"
        )

        plt.plot(
            df2["frame"],
            df2[feature],
            label="Crowding"
        )

        plt.xlabel("Frame")

        plt.ylabel(feature)

        plt.title(feature)

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()