from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms


class G_model(nn.Module):
    def __init__(self):
        # add flatten here
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 784 * 2),
            nn.LeakyReLU(),
            nn.Linear(1568, 784)
        )

    def forward(self, X):
        return self.model(X)


class D_model(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=2),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels=10, out_channels=128,
                      kernel_size=2, stride=2),
            nn.AvgPool2d(3),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes),
        )

    def forward(self, X):
        return self.model(X)


class GAN(nn.Module):
    # Just concat initial picture with noise?
    def __init__(self, n_classes):
        super().__init__()
        self.G = G_model()
        self.D = D_model(n_classes)

    def forward(self):
        pass


class GAN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.G = G_model()
        self.D = D_model()

    def train(self):
        # not clear where it should be, I think it should be in forward, somehow
        criterion = nn.BCEWithLogitsLoss()

        optimizer_G = torch.optim.SGD(self.G.parameters(), lr=0.01, momentum=0.9)
        optimizer_D = torch.optim.SGD(self.D.parameters(), lr=0.01, momentum=0.9)

        noise = torch.randn(size=(32, len(X), 28, 28))
        zeros = torch.zeros(len(X))
        ones = torch.ones(len(X))

        EPOCHES = 10
        batch_size = 32
        dataloader = [] # TODO

        for _ in range(EPOCHES):
            for X, _ in dataloader:

                # iteration for discriminator
                z = self.G(noise)
                pred_fake = self.D(z.detach())
                pred_real = self.D(X)

                D_loss = criterion(pred_fake, zeros) + criterion(pred_real, ones)
                D_loss.backward()

                optimizer_D.step()

                # iteration for the generator
                pred = self.D(z)
                G_loss = criterion(pred, ones)
                G_loss.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()
                optimizer_G.zero_grad()


class CGAN(nn.Module):
    pass


if __name__ == "__main__":
    pass
