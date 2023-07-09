from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


class G_model(nn.Module):
    def __init__(self):
        # add flatten here
        super().__init__()
        self.unflatten = nn.Unflatten(-1, (28, 28))
        self.model = nn.Sequential(
            nn.Linear(784, 784 * 2),
            nn.LeakyReLU(),
            nn.Linear(1568, 784)
        )

    def forward(self, X):
        return self.unflatten(self.model(torch.flatten(X, -2, -1)))

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


# class GAN(nn.Module):
#     def __init__(self, n_classes):
#         super().__init__()
#         self.G = G_model()
#         self.D = D_model()

#     def train(self):
#         # not clear where it should be, I think it should be in forward, somehow
#         criterion = nn.BCEWithLogitsLoss()

#         optimizer_G = torch.optim.SGD(
#             self.G.parameters(), lr=0.01, momentum=0.9)
#         optimizer_D = torch.optim.SGD(
#             self.D.parameters(), lr=0.01, momentum=0.9)

#         noise = torch.randn(size=(32, len(X), 28, 28))
#         zeros = torch.zeros(len(X))
#         ones = torch.ones(len(X))

#         EPOCHES = 10
#         dataloader = []  # TODO

#         for _ in range(EPOCHES):
#             for X, _ in dataloader:

#                 # iteration for discriminator
#                 z = self.G(noise)
#                 pred_fake = self.D(z.detach())
#                 pred_real = self.D(X)

#                 D_loss = criterion(pred_fake, zeros) + \
#                     criterion(pred_real, ones)
#                 D_loss.backward()

#                 optimizer_D.step()

#                 # iteration for the generator
#                 pred = self.D(z)
#                 G_loss = criterion(pred, ones)
#                 G_loss.backward()
#                 optimizer_G.step()

#                 optimizer_D.zero_grad()
#                 optimizer_G.zero_grad()

class GAN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.G = G_model()
        self.D = D_model(n_classes)

    def train(self, dataloader):
        # not clear where it should be, I think it should be in forward, somehow
        criterion = nn.BCEWithLogitsLoss()

        optimizer_G = torch.optim.SGD(
            self.G.parameters(), lr=0.01, momentum=0.9)
        optimizer_D = torch.optim.SGD(
            self.D.parameters(), lr=0.01, momentum=0.9)
        
        batch_size = dataloader.batch_size

        zeros = torch.zeros(batch_size)
        ones = torch.ones(batch_size)

        EPOCHES = 10
        G_loss_lst, D_loss_lst = [], []


        for _ in tqdm(range(EPOCHES)):
            G_running_loss = 0
            D_running_loss = 0

            for X, _ in dataloader:
                noise = torch.randn(size=(batch_size, 1, 28, 28))
                # iteration for discriminator
                z = self.G(noise)
                pred_fake = self.D(z.detach()).squeeze(-1)
                pred_real = self.D(X).squeeze(-1)

                D_loss = criterion(pred_fake, zeros) + \
                    criterion(pred_real, ones)
                D_loss.backward()
                optimizer_D.step()
                D_running_loss += D_loss.item()

                # iteration for the generator
                pred = self.D(z).squeeze(-1)
                G_loss = criterion(pred, ones)


                G_loss.backward()
                optimizer_G.step()
                G_running_loss += G_loss.item()

                optimizer_D.zero_grad()
                optimizer_G.zero_grad()
                print(D_loss)
            G_loss_lst.append(G_running_loss)
            D_loss_lst.append(D_running_loss)

        return G_loss_lst, D_loss_lst

if __name__ == "__main__":
    pass
