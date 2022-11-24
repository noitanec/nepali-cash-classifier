import random
import glob

from torch.utils.data import Dataset
from PIL import Image

idxtolabel = {
    0: "five",
    1: "twenty",
    2: "fivehundred",
    3: "fifty",
    4: "ten",
    5: "thousand",
    6: "hundred",
}
labeltoidx = {v: k for k, v in idxtolabel.items()}


class CustomDataset(Dataset):
    """This is a custom dataset class for nepali cash dataset publicly
    available in https://drive.google.com/file/d/1pDF0hx6pvgx4DJTCHL4EeDdCT4wlfnGW/view"""

    def __init__(self, img_dir, transform=None, k=None):
        self.transform = transform
        self.data = []

        classpath = glob.glob(f"{img_dir}/*")
        for cls in classpath:
            label = labeltoidx[cls.split("/")[-1]]
            for img_path in glob.glob(cls + "/*.jpg"):
                self.data.append((img_path, label))

        if k:
            self.data = random.sample(self.data, k)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label
