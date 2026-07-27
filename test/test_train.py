from src.training.trainer import Trainer

trainer = Trainer(
    dataset_dir="datasets/sequences", batch_size=8, epochs=30, learning_rate=0.001
)

trainer.train()
