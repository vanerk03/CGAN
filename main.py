from torch import nn
import torch
import torchvision.transforms as T
from tqdm.notebook import tqdm
import os
import torch
from torch.utils.data import Dataset
from IPython.display import display

from models.unet import UNet
from torchvision.io import read_image

class FacadesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = self._get_filenames(root_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # image_path = os.path.join(self.root_dir, self.image_filenames[idx])
        img_path = os.path.join(self.root_dir, self.image_filenames[idx])
        
        image = read_image(img_path).to(torch.float32)
        image = self.transform(image)

        image1 = image[:, :256, :256]
        image2 = image[:, :256, 256:512]

        return image1, image2

    def _get_filenames(self, directory):
        filenames = sorted(os.listdir(directory))
        return filenames


if __name__ == "__main__":
    transform = T.Compose([
        T.Lambda(lambda x: x / 255),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    inverse_transform = T.Normalize(mean=(-1, -1, -1), std=(2, 2, 2))


    # =========================
    # parameters
    # =========================

    EPOCHS = 10
    batch_size = 32
    model = UNet()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)



    data_dir = './data/facades/train/'
    dataset = FacadesDataset(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    iters = 5
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        running_loss = 0
        
        if iters == 0:
            break
        
        for X, y in tqdm(dataloader, leave=False, desc=f"Epoch {epoch}"):
            
            if iters == 0:
                break
            
            pred = model(X)

            loss = criterion(pred, y)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            
            # for profiling
            iters -= 1
                
        print(f"epoch {epoch}: loss={running_loss}")
        break