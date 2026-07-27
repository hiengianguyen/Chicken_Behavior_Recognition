import torch
import torch.nn as nn


class ChickenBehaviorLSTM(nn.Module):

    def __init__(
        self, input_size=9, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        logits = self.classifier(last_hidden)

        return logits
