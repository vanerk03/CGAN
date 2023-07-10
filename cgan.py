from torch import nn
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import tqdm


class Reshape(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, *args):
        super().__init__()
        self.dims = args

    def forward(self, input):
        return input.view(input.size(0), *self.dims)


class G_model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.conditional = nn.Sequential(
            nn.Linear(50, 49),
            Reshape(1, 7, 7)
        )

        self.reshape_noise = nn.Sequential(
            nn.Linear(100, 6272),
            nn.LeakyReLU(0.2),
            Reshape(128, 7, 7),
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(129, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, X, y):
        # create embedding for y
        ohe = torch.eye(50, device=self.device)[y]
        embed = self.conditional(ohe)
        noise = self.reshape_noise(X)
        return self.model(torch.cat((embed, noise), dim=1))


class D_model(nn.Module):
    def __init__(self, n_classes, device):
        super().__init__()
        self.device = device
        self.conditional = nn.Sequential(
            nn.Linear(50, 784),
            Reshape(1, 28, 28)
        )

        self.model = nn.Sequential(
            # two in_channels after concat
            nn.Conv2d(in_channels=2, out_channels=128,
                      kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(6272, n_classes)
        )

    def forward(self, X, y):
        # create embedding for y
        ohe = torch.eye(50, device=self.device)[y]
        embed = self.conditional(ohe)
        return self.model(torch.cat((embed, X), dim=1))


class CGAN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # terrible thing with device, mb this this
        self.G = G_model(device).to(device)
        self.D = D_model(1, device).to(device)

    def forward(self, y):
        noise = torch.randn((len(y), 100), device=self.device)
        return self.G(noise)

    def train(self, dataloader, EPOCHS, verbose=True):
        noise_fixed = torch.randn((5, 100), device=self.device)


        criterion = nn.BCEWithLogitsLoss()

        optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=3e-4)
        optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=3e-4)


        batch_size = dataloader.batch_size
        G_loss_lst, D_loss_lst = [], []

        for epoch in tqdm.notebook.tqdm(range(EPOCHS), desc="Epoch"):

            G_running_loss = 0
            D_running_loss = 0
            for i, (X, y) in enumerate(tqdm.notebook.tqdm(dataloader, leave=False, desc=f"Epoch {epoch}")):
                # Label Smoothing,
                # i.e. if you have two target labels:
                # Real=1 and Fake=0, then for each incoming sample, if it is real,
                # then replace the label with a random number between 0.7 and 1.2,
                # and if it is a fake sample, replace it with 0.0 and 0.3 (for example).
                # Salimans et. al. 2016
                ones = (0.7 - 1.2) * torch.rand(batch_size,
                                                device=self.device) + 1.2
                zeros = 0.3 * torch.rand(batch_size, device=self.device)

                noise = torch.randn(size=(batch_size, 100), device=self.device)
                # iteration for discriminator
                z = self.G(noise, y)

                pred_fake = self.D(z.detach(), y).squeeze(-1)
                pred_real = self.D(X.to(self.device), y).squeeze(-1)

                D_loss = criterion(pred_fake, zeros) + \
                    criterion(pred_real, ones)

                D_loss.backward()
                optimizer_D.step()
                D_running_loss += D_loss.item() / batch_size

                # iteration for the generator
                pred = self.D(z, y).squeeze(-1)
                G_loss = criterion(pred, ones)

                G_loss.backward()
                optimizer_G.step()
                G_running_loss += G_loss.item() / batch_size

                optimizer_D.zero_grad()
                optimizer_G.zero_grad()

                # fix noise + logging
                if verbose and (i + 1) % 500 == 0:
                    with torch.no_grad():
                        print(
                            f"Epoch {_+1}, batch {i+1}, D_loss: {D_loss.item()}, G_loss: {G_loss.item()}")
                        num_imgs = 5  # Adjust the number of columns as per your preference
                        # Adjust the figsize as per your preference
                        _, axs = plt.subplots(1, num_imgs, figsize=(15, 5))

                        # imgs = self.G(noise_fixed)
                        imgs = self.G(noise_fixed)

                        for j, img in enumerate(imgs):
                            axs[j].imshow(
                                img.cpu().detach().numpy().squeeze(), cmap="gray")

                        plt.show()

            G_loss_lst.append(G_running_loss)
            D_loss_lst.append(D_running_loss)

            print(G_running_loss, D_running_loss)

        return G_loss_lst, D_loss_lst

    def D_step(self, X, y):
        pass

    def G_step(self, X, y):
        pass


if __name__ == "__main__":
    device = "mps:0" if getattr(torch, 'has_mps', False) \
        else "cuda:0" if torch.cuda.is_available() else "cpu"

    model = G_model(device)
    X = torch.randn((32, 100))
    y = torch.randint(0, 9, (32,))

    z = model(X, y)
    discriminator = D_model(1, device)

    print(discriminator(z, y).shape)
