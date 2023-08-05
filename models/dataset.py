import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import os
import torchvision.transforms as T


def rnd_horizontal_flip(batch, p, fixed_param=0.5):
    # fixed param - prob of flipping
    # assert len(batch.shape) == 4, "batch should have 4 dimensions"
    if p < fixed_param:
        return torch.flip(batch, dims=[-1])
    else:
        return batch


class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, flip=True, resize_crop=True):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = self._get_filenames(root_dir)
        self.flip = flip
        self.resize_crop = resize_crop

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # image_path = os.path.join(self.root_dir, self.image_filenames[idx])
        img_path = os.path.join(self.root_dir, self.image_filenames[idx])

        image = read_image(img_path).to(torch.float32)

        new_width = image.shape[-1] // 2
        
        image1 = image[:, :, :new_width]
        image2 = image[:, :, new_width:]

        # NOTE: if statements might slow down the training
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        if self.resize_crop:
            i, j, h, w = T.RandomCrop.get_params(
                image1, output_size=(256, 256)
            )

            image1 = TF.crop(image1, i, j, h, w)
            image2 = TF.crop(image2, i, j, h, w)

        if self.flip:
            random_number = torch.rand(1)
            image1 = rnd_horizontal_flip(image1, random_number)
            image2 = rnd_horizontal_flip(image2, random_number)

        # NOTE: change later, swap image2 and image1 everywhere
        return image2, image1

    def _get_filenames(self, directory):
        filenames = sorted(os.listdir(directory))
        return filenames
