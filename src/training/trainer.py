from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.sequence.dataset import ChickenDataset
from src.models.lstm import ChickenBehaviorLSTM


class Trainer:

    def __init__(self, dataset_dir, batch_size=8, epochs=30, learning_rate=0.001):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\nUsing Device : {self.device}")

        # Dataset
        dataset = ChickenDataset(dataset_dir)

        train_size = int(len(dataset) * 0.8)

        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model
        self.model = ChickenBehaviorLSTM(num_classes=2).to(self.device)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.epochs = epochs

        self.best_accuracy = 0

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0

        correct = 0

        total = 0

        for X, y in self.train_loader:

            X = X.to(self.device)

            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(X)

            loss = self.criterion(outputs, y)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            prediction = torch.argmax(outputs, dim=1)

            correct += (prediction == y).sum().item()

            total += y.size(0)

        accuracy = correct / total

        loss = total_loss / len(self.train_loader)

        return loss, accuracy

    def validate(self):

        self.model.eval()

        total_loss = 0

        correct = 0

        total = 0

        with torch.no_grad():

            for X, y in self.val_loader:

                X = X.to(self.device)

                y = y.to(self.device)

                outputs = self.model(X)

                loss = self.criterion(outputs, y)

                total_loss += loss.item()

                prediction = torch.argmax(outputs, dim=1)

                correct += (prediction == y).sum().item()

                total += y.size(0)

        accuracy = correct / total

        loss = total_loss / len(self.val_loader)

        return loss, accuracy

    def save_model(self):

        Path("weights").mkdir(exist_ok=True)

        torch.save(self.model.state_dict(), "weights/best_model.pt")

    def train(self):

        print("\nStart Training...\n")

        for epoch in range(self.epochs):

            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.validate()

            print(
                f"Epoch {epoch+1}/{self.epochs}"
                f" | Train Loss : {train_loss:.4f}"
                f" | Train Acc : {train_acc:.3f}"
                f" | Val Loss : {val_loss:.4f}"
                f" | Val Acc : {val_acc:.3f}"
            )

            if val_acc > self.best_accuracy:

                self.best_accuracy = val_acc

                self.save_model()

                print("Best Model Saved!")

        print("\nTraining Finished!")
