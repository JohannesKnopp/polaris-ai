import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 16)
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        # self.decoder = torch.nn.Sequential(
        #     nn.Linear(9, 18),
        #     nn.ReLU(),
        #     nn.Linear(18, 36),
        #     nn.ReLU(),
        #     nn.Linear(36, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 28 * 28),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        return encoded