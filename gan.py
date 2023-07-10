from torch import nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import sys
import platform
from IPython.display import clear_output
from tqdm.notebook import tqdm
from time import sleep


class Reshape(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, *args):
        super().__init__()
        self.dims = args

    def forward(self, input):
        return input.view(input.size(0), *self.dims)

class Reshape28(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(input.shape[0], 1, 28, 28)


class G_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            Reshape28(),
            nn.Tanh(),
        )

    def forward(self, X):
        return self.model(X)

class D_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 28 * 28),
            nn.LeakyReLU(0.2),
            nn.Linear(28 * 28, 1),
            nn.LeakyReLU(0.2),
        )


    def forward(self, X):
        X_ = X.view(batch_size, -1)
        return self.model(X_)

class GAN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # terrible thing with device, mb this this
        self.G = G_model().to(device)
        self.D = D_model(1).to(device)

    def forward(self, n):
        noise = torch.randn((n, 100), device=self.device)
        return self.G(noise)

    def train_(self, dataloader, EPOCHS=100, verbose=True):
        num_imgs = 15
        noise_fixed = torch.randn((num_imgs, 100), device=self.device)

        criterion = nn.BCEWithLogitsLoss()

        optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=3e-4)
        optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=3e-4)


        batch_size = dataloader.batch_size
        G_loss_lst, D_loss_lst = [], []

        for epoch in tqdm(range(EPOCHS), desc="Epoch"):
            G_running_loss = 0
            D_running_loss = 0
            for i, (X, _) in enumerate(tqdm(dataloader, leave=False, desc=f"Epoch {epoch}")):
                # Label Smoothing,
                # i.e. if you have two target labels:
                # Real=1 and Fake=0, then for each incoming sample, if it is real,
                # then replace the label with a random number between 0.7 and 1.2,
                # and if it is a fake sample, replace it with 0.0 and 0.3 (for example).
                # Salimans et. al. 2016
                ones = (0.7 - 1.2) * torch.rand(batch_size,
                                                device=self.device) + 1.2
                zeros = 0.3 * torch.rand(batch_size, device=self.device)


                # ones = torch.ones(batch_size, device=self.device)
                # zeros = torch.zeros(batch_size, device=self.device)

                noise = torch.randn((batch_size, 100), device=self.device)

                # iteration for discriminator
                z = self.G(noise)

                pred_fake = self.D(z.detach()).squeeze(-1)
                pred_real = self.D(X.to(self.device)).squeeze(-1)

                D_loss = criterion(pred_fake, zeros) + \
                    criterion(pred_real, ones)

                D_loss.backward()
                optimizer_D.step()
                D_running_loss += D_loss.item() / batch_size

                # iteration for the generator
                pred = self.D(z).squeeze(-1)
                G_loss = criterion(pred, ones)

                G_loss.backward()
                optimizer_G.step()
                G_running_loss += G_loss.item() / batch_size

                optimizer_D.zero_grad()
                optimizer_G.zero_grad()


                # fix noise + logging
                if i % 500 == 0:
                    self.eval()
                    with torch.no_grad():
                        # Adjust the figsize as per your preference
                        _, axs = plt.subplots(1, num_imgs, figsize=(17, 6))

                        # imgs = self.G(noise_fixed)
                        imgs = self.G(noise_fixed)
                        for j, img in enumerate(imgs):
                            axs[j].imshow(
                                denorm(img.cpu().detach()).numpy().squeeze(), cmap="gray")

                        plt.show()
                        self.train()

            G_loss_lst.append(G_running_loss)
            D_loss_lst.append(D_running_loss)

            print("G_loss:", G_running_loss, "D_loss:", D_running_loss)

        return G_loss_lst, D_loss_lst
