import torch
import os
from torch import nn


class Pix2PixCGAN(nn.Module):
    def __init__(self, D_model, G_model, device='cpu'):
        super().__init__()
        self.device = device

        self.G = G_model.to(device)
        self.D = D_model.to(device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.lmbda = 100

        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))

        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # number of finished epoches
        self.epoch = 0

    def forward(self, X):
        X = X.to(self.device)
        if len(X) == 1:
            return self.G(X)

        res = []
        for i in range(len(X)):
            res.append(self.G(X[i].unsqueeze(0)))
        return torch.cat(res)

    def D_step(self, X, y):
        """
        given X and y
        y is the desired output of G
        X is an input
        z is a real output of G
        """
        # ones_noisy = (0.7 - 1.2) * torch.rand(batch_size,
        #                                 device=self.device) + 1.2
        # zeros_noisy = 0.3 * torch.rand(batch_size, device=self.device)

        self.optimizer_D.zero_grad()
        y_cat = torch.cat([y, X], dim=1)

        z = self.G(X)
        z_cat = torch.cat([z, X], dim=1)

        pred_fake = self.D(z_cat.detach())
        pred_real = self.D(y_cat)

        ones = torch.ones_like(pred_fake, device=self.device)
        zeros = torch.zeros_like(pred_fake, device=self.device)

        D_loss = (self.criterion(pred_fake, zeros) +
                  self.criterion(pred_real, ones)) / 2

        D_loss.backward()
        self.optimizer_D.step()

        return z_cat, z, D_loss.item()

    def G_step(self, z_cat, z, y):
        # iteration for the generator
        self.optimizer_G.zero_grad()

        pred = self.D(z_cat)

        ones = torch.ones_like(pred, device=self.device)

        GAN_G_loss = self.criterion(pred, ones)
        l1_G_loss = self.l1_loss(z, y) * self.lmbda
        G_loss = GAN_G_loss + l1_G_loss

        G_loss.backward()
        self.optimizer_G.step()

        return G_loss.item()

    def save_checkpoint(self, PATH):
        PATH = os.path.join(PATH, f"patchgan_{self.epoch}.pth")
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }, PATH)

    def load_checkpoint(self, PATH):
        checkpoint = torch.load(PATH, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.epoch = checkpoint["epoch"]