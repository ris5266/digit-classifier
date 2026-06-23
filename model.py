import torch
import torch.nn as nn


class DigitClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DigitClassifier()

    # sanity check
    dummy_input = torch.randn(8, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
