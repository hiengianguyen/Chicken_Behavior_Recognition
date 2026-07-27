from torch.utils.data import DataLoader

from src.sequence.dataset import ChickenDataset

dataset = ChickenDataset("datasets/sequences")

print()

print("Dataset Length:", len(dataset))

x, y = dataset[0]

print()

print("One Sample")

print("x shape:", x.shape)

print("label:", y)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

print()

print("Testing DataLoader")

for batch_x, batch_y in loader:

    print(batch_x.shape)

    print(batch_y.shape)

    break
