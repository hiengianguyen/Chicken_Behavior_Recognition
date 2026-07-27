from pathlib import Path

import numpy as np
import pandas as pd


class SequenceBuilder:

    def __init__(self,
                 window_size=90,
                 stride=15):

        self.window_size = window_size
        self.stride = stride

    def _load_feature_file(self, csv_path):

        df = pd.read_csv(csv_path)

        return df
    
    def _build_sequences(self, df, label):

        sequences = []

        labels = []

        # Bỏ cột frame
        features = df.drop(columns=["frame"]).values

        total = len(features)

        for start in range(
                0,
                total - self.window_size + 1,
                self.stride):

            end = start + self.window_size

            sequence = features[start:end]

            sequences.append(sequence)

            labels.append(label)

        return sequences, labels
    
    def build(self, feature_files):

        X = []

        y = []

        for csv_path, label in feature_files:

            df = self._load_feature_file(csv_path)

            sequences, labels = self._build_sequences(
                df,
                label
            )

            X.extend(sequences)

            y.extend(labels)

        X = np.array(X, dtype=np.float32)

        y = np.array(y, dtype=np.int64)

        return X, y
    
    def save(self, X, y, output_dir):

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            output_dir / "X.npy",
            X
        )

        np.save(
            output_dir / "y.npy",
            y
        )

        print()

        print("Dataset Saved")

        print("X:", X.shape)

        print("y:", y.shape)