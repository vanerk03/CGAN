from torch import nn
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import sys
import platform
from IPython.display import clear_output
from tqdm.notebook import tqdm
import numpy as np


class G_model(nn.Module):
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

    def forward(self, X, labels):
        emb = self.embed(labels)
        return self.model(torch.cat((X, emb), dim=1)).view(-1, 28, 28)

class D_model(nn.Module):
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


    def forward(self, X, labels):
        emb = self.embed(labels)
        X_ = X.view(len(labels), -1)
        return self.model(torch.cat((X_, emb), dim=1))

class CGAN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # terrible thing with device, mb this this
        self.G = G_model().to(device)
        self.D = D_model().to(device)

    def forward(self, n, labels=None):
        if labels is None:
            labels = torch.randint(0, 10, (n,)).to(self.device)
        labels = labels.to(self.device)
        noise = torch.randn((n, 100), device=self.device)
        return self.G(noise, labels)

    def train_(self, dataloader, EPOCHS=100, verbose=True):
        num_imgs = 10
        noise_fixed = torch.randn((num_imgs, 100), device=self.device)
        labels_fixed = torch.arange(10).to(self.device)

        criterion = nn.BCEWithLogitsLoss()

        optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=3e-4)
        optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=3e-4)


        batch_size = dataloader.batch_size
        G_loss_lst, D_loss_lst = [], []

        for epoch in tqdm(range(EPOCHS), desc="Epoch"):
            G_running_loss_lst = []
            D_running_loss_lst = []
            for i, (X, labels) in enumerate(tqdm(dataloader, leave=False, desc=f"Epoch {epoch}")):
                # Label Smoothing,
                # i.e. if you have two target labels:
                # Real=1 and Fake=0, then for each incoming sample, if it is real,
                # then replace the label with a random number between 0.7 and 1.2,
                # and if it is a fake sample, replace it with 0.0 and 0.3 (for example).
                # Salimans et. al. 2016
                # ones = (0.7 - 1.2) * torch.rand(batch_size,
                                                # device=self.device) + 1.2
                # zeros = 0.3 * torch.rand(batch_size, device=self.device)


                ones = torch.ones(batch_size, device=self.device)
                zeros = torch.zeros(batch_size, device=self.device)

                noise = torch.randn((batch_size, 100), device=self.device)
                labels = labels.to(self.device)

                # iteration for discriminator
                z = self.G(noise, labels)

                pred_fake = self.D(z.detach(), labels).squeeze(-1)
                pred_real = self.D(X.to(self.device), labels).squeeze(-1)

                D_loss = criterion(pred_fake, zeros) + \
                    criterion(pred_real, ones)

                D_loss.backward()
                optimizer_D.step()
                D_running_loss_lst.append(D_loss.item())

                # iteration for the generator
                pred = self.D(z, labels).squeeze(-1)
                G_loss = criterion(pred, ones)

                G_loss.backward()
                optimizer_G.step()
                G_running_loss_lst.append(G_loss.item())

                optimizer_D.zero_grad()
                optimizer_G.zero_grad()


                # fix noise + logging
                if (i + 1) % 500 == 0:
                    self.eval()
                    with torch.no_grad():
                        # Adjust the figsize as per your preference
                        _, axs = plt.subplots(1, num_imgs, figsize=(17, 6))

                        # imgs = self.G(noise_fixed)
                        imgs = self.G(noise_fixed, labels_fixed)
                        for j, img in enumerate(imgs):
                            axs[j].imshow(
                                undo_transform(img.cpu().detach().view(1, 28, 28)).view(28, 28).numpy(), cmap="gray")

                        plt.show()
                        self.train()

                        print("G_loss:", np.mean(G_running_loss_lst), "D_loss:", np.mean(D_running_loss_lst))
                        
                        plt.plot(torch.tensor(G_running_loss_lst).view(-1, 10).mean(1), label='G-loss')
                        plt.plot(torch.tensor(D_running_loss_lst).view(-1, 10).mean(1), label='D-loss')
                        plt.legend(loc='upper right')

                        plt.show()

            G_loss_lst.append(np.mean(G_running_loss_lst))
            D_loss_lst.append(np.mean(D_running_loss_lst))

            print("G_loss:", np.mean(G_running_loss_lst), "D_loss:", np.mean(D_running_loss_lst))
            
            plt.plot(G_running_loss_lst, label='G-loss')
            plt.plot(D_running_loss_lst, label='D-loss')
            plt.legend(loc='upper right')

            plt.show()

        return G_loss_lst, D_loss_lst
