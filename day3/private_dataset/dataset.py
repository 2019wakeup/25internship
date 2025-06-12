import os
from torch.utils.data import Dataset
from PIL import Image

class ImageTxtDataset(Dataset):
    def __init__(self, txt_path: str, folder_name, transform):
        self.transform = transform
        self.data_dir = os.path.dirname(txt_path)
        self.imgs_path = []
        self.labels = []
        self.folder_name = folder_name

        # 读取txt文件
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        # 读取txt文件中的每一行
        for line in lines:
            img_path, label = line.split()
            label = int(label.strip())
            #img_path = os.path.join(self.data_dir, self.folder_name, img_path)
            self.labels.append(label)
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, i):
        path, label = self.imgs_path[i], self.labels[i]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label