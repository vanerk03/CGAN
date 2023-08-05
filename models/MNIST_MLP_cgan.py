import torch
import os
from torch import nn

class MLPGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
        )

    def forward(self, X, y):
        emb = self.embed(y)
        return self.model(torch.cat((X, emb), dim=1)).view(-1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(784 + 10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 28 * 28),
            nn.LeakyReLU(0.2),
            nn.Linear(28 * 28, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, X, y):
        emb = self.embed(y)
        X_ = X.view(len(X), -1)
        return self.model(torch.cat((X_, emb), dim=1))


class MnistCGAN(nn.Module):

    def __init__(self, D_model, G_model, device='cpu'):
        super().__init__()

        self.noise_size = 100

        self.device = device

        self.G = G_model.to(device)
        self.D = D_model.to(device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=3e-4, betas=(0.9, 0.999))

        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=3e-4, betas=(0.9, 0.999))

    def forward(self, n, labels=None):
        if labels is None:
            labels = torch.randint(0, 10, (n,), device=self.device)
        else:
            labels = labels.to(self.device)

        noise = torch.randn((n, self.noise_size), device=self.device)
        return self.G(noise, labels).view(n, 1, 28, 28)


    def D_step(self, X, y):
        self.optimizer_D.zero_grad()

        noise = torch.randn((len(y), self.noise_size), device=self.device)

        z = self.G(noise, y)

        pred_fake = self.D(z.detach(), y)
        pred_real = self.D(X, y)

        ones = torch.ones_like(pred_fake, device=self.device)
        zeros = torch.zeros_like(pred_fake, device=self.device)

        D_loss = (self.criterion(pred_fake, zeros) + \
            self.criterion(pred_real, ones)) / 2

        D_loss.backward()
        self.optimizer_D.step()
        return z, D_loss.item()

    def G_step(self, z, y):
        # iteration for the generator
        self.optimizer_G.zero_grad()

        pred = self.D(z, y)

        ones = torch.ones_like(pred, device=self.device)

        G_loss = self.criterion(pred, ones)

        G_loss.backward()
        self.optimizer_G.step()

        return G_loss.item()
