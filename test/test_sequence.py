from src.sequence.builder import SequenceBuilder

builder = SequenceBuilder(window_size=90, stride=15)

X, y = builder.build(
    [
        ("datasets/behavior/b_1.csv", 0),
        ("datasets/behavior/b_2.csv", 0),
        ("datasets/behavior/b_3.csv", 0),
        ("datasets/behavior/b_4.csv", 0),
        ("datasets/behavior/b_5.csv", 0),
        ("datasets/behavior/b_6.csv", 0),
        ("datasets/behavior/normal.csv", 0),
        ("datasets/behavior/standing.csv", 1),
    ]
)

builder.save(X, y, "datasets/sequences")
