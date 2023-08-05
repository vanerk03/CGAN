import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import wandb

def display_images(imgs, title="input, output, ground truth", cmap=None, figsize=(11, 6)):
    """
    imgs: list of tensors = [tensor_with_imgs, ...]
        then 2d grid of images
    imgs: tensor
        then assume tensor.shape = (batch_size, channels, width, height)
    """
    if type(imgs) == torch.Tensor:
        fig, ax = plt.subplots(nrows=1, ncols=len(imgs), figsize=figsize)
        for i in range(len(imgs)):
            ax[i].imshow(imgs[i].numpy(), cmap=cmap)
            ax[i].set_xticks([])
            ax[i].set_yticks([])

    else:
        num_imgs = len(imgs[0])
        fig, ax = plt.subplots(nrows=len(imgs), ncols=num_imgs, figsize=figsize)

        for i in range(len(imgs)):
            for j in range(num_imgs):
                ax[i, j].imshow(imgs[i][j].numpy(), cmap=cmap)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

    plt.suptitle(title)
    return fig



def log_imgs(model, loader, inverse_transform, local=True, step=None, num_imgs=5):
    # local: bool - if true: locally
    # else: use wandb logger

    with torch.no_grad():
        batch = next(iter(loader))

        X, y = batch
        pred = [None] * num_imgs

        pred = model(X).cpu()
        pred_imgs = inverse_transform(pred).permute(0, 2, 3, 1)
        
        truth = inverse_transform(y).permute(0, 2, 3, 1)
        X = inverse_transform(X).permute(0, 2, 3, 1)

        fig = display_images(
            [X[:num_imgs], pred_imgs[:num_imgs], truth[:num_imgs]]
        )

        if local:
            plt.show()
        else:
            wandb.log({"Image validation": fig}, step=step)
            fig.close()


def init_weights(model, mean=0.0, std=0.02):
    for name, m in model._modules.items():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, mean, std)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, std)
            nn.init.constant_(m.bias.data, 0.0)
        else:
            init_weights(m, mean, std)
