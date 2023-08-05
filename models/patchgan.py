from .unet import ConvBlock
from torch import nn


class PatchGAN(nn.Module):
    def __init__(self, in_channels=6, features=[64, 128, 256, 512]):
        super().__init__()
        model = []
        print("=========| PatchGAN initialized |===========")
        for i, feature in enumerate(features):
            batch_norm = True
            if i == 0:
                batch_norm = False

            print(f"log in/out channels:\nin: {in_channels} out: {feature}")
            model.append(ConvBlock(
                in_channels=in_channels,
                out_channels=feature,
                dropout=False,
                activation='leaky',
                batch_norm=batch_norm,
                downsample=True,
                conv_args={"stride": 1 if i == len(features) - 1 else 2}
            )
            )
            in_channels = feature

        model.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=1
        ))

        print("=" * 30)

        self.model = nn.Sequential(*model)

    def forward(self, X):
        return self.model(X)
