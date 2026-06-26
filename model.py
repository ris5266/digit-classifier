import torch
import torch.nn as nn


# connected net for the 28x28 digits
class DigitClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 10),
        )

    def forward(self, x):
        # flatten image then run it through the layers
        x = self.flatten(x)
        x = self.network(x)
        return x


if __name__ == "__main__":
    model = DigitClassifier()

    # check that the shapes work
    dummy = torch.randn(8, 1, 28, 28)
    out = model(dummy)
    print(out.shape)
