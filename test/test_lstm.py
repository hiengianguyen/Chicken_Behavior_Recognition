import torch

from src.models.lstm import ChickenBehaviorLSTM

# Khởi tạo model
model = ChickenBehaviorLSTM()

# In kiến trúc
print(model)

print("-" * 50)

# Tạo dữ liệu giả
x = torch.randn(8, 90, 18)  # Batch size  # Sequence length  # Feature

print("Input Shape :", x.shape)

# Forward
output = model(x)

print("Output Shape:", output.shape)

print(output)

print("-" * 50)

print("Prediction")

pred = torch.argmax(output, dim=1)

print(pred)
