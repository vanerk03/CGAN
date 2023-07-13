from torch import nn
import torch

# =========================================
# UNet class
# =========================================

class UNet(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 512, 512, 512, 512]):
        # TODO: take into account picture size, crop/resize

        super().__init__()
        # mb add initialization based on our blocks sizes?
        # potential problem with Device
        # dropout should not be in eval mode
        # same for the batchnorm, which is instance norm in our case

        # C64-C128-C256-C512-C512-C512-C512-C512
        self.encoder = nn.ModuleList()

        # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
        self.decoder = nn.ModuleList()


        for i, feature in enumerate(features):
            batch_norm = True
            dropout = False
            activation = "leaky"

            # innermost
            if i == 0:
                batch_norm = False

            # bottleneck
            if i == len(features) - 1:
                batch_norm = False
                activation = "relu"

            self.encoder.append(ConvBlock(
                in_channels=in_channels,
                out_channels=feature,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm,
                downsample=True)
            )

            in_channels = feature

        # NOTE WTF WITH DIMENTIONS?????????????????

        # decoder
        for i in range(len(features)):
            feature = features[- i - 1]
            batch_norm = True
            dropout = True
            activation = "relu"
            in_channels_ = feature * 2
            
            if i == len(features) - 1:
                out_channels_ = 3
            else:
                out_channels_ = features[- i - 2]

            # bottleneck does not have skip connection
            if i == 0:
                in_channels_ = feature
                out_channels_ = feature

            print(f"{i} in:", in_channels_, "out:", out_channels_)

            self.decoder.append(ConvBlock(
                in_channels=in_channels_,
                out_channels=out_channels_,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm,
                downsample=False)
            )


        # TODO: check the last dimension: https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
        # (None, 128, 256, 256)
        # is number of filters the NC after or before?
        # in the innermost layer it says nc = 64
        # but in the outmost it also says 64, why is it 3 at the end then?

        self.activation = nn.Tanh()

    def forward(self, X):
        n = len(self.encoder)

        residual = [None] * n

        # encoder pass
        for i in range(len(self.encoder)):
            X = self.encoder[i](X)
            # print(i, X.shape)
            residual[i] = X

        # print("=" * 20)
        # decoder pass
        for i in range(len(self.decoder)):
            # bottleneck no skip connection
            if i != 0:
                # print("shape before cat", X.shape)
                # print(f"residual: {n - i - 1}, {i}", residual[n - i - 1].shape)
                X = torch.cat((X, residual[n - i - 1]), dim=1)

            # print(f"{i} in: ", X.shape)
            X = self.decoder[i](X)
            # print(f"{i} out: ", X.shape)
            # print("-" * 20)

        return self.activation(X)
    
# =========================================
# Helper classes
# =========================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation: str, batch_norm=True, dropout=False, downsample=True):
        super().__init__()
        model = []

        # TODO: bias = False?
        if downsample:
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        else:
            conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )

        model.append(conv)

        match activation:
            case "relu":
                model.append(nn.ReLU())
            case "leaky":
                model.append(nn.LeakyReLU(0.2))
            case _:
                raise ValueError("Invalid activation, choose: 'relu', 'leaky'")

        if batch_norm:
            model.append(nn.BatchNorm2d(out_channels))

        if dropout:
            model.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*model)

    def forward(self, X):
        return self.model(X)
