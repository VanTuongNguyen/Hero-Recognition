
from torch.utils.data import Dataset
import torch
import numpy as np
import glob

class CustomDataSet(Dataset):

    def __init__(self,data_path,
                transform=None):

        self.transform = transform

        #since the data is not much, we can load it
        #entirely in RAM
        files_path = glob.glob(f'{data_path}/*.npz')

        image = []
        name = []
        for path in files_path:
            data = np.load(path)
            image.append(data["image"])
            name.append(data["name"])
        

        image = np.concatenate(image,0)
        name = np.concatenate(name,0)

       
        self.x_data = np.array(image)
        self.y_data = np.array(name)


        print('x (images) shape: ',self.x_data.shape)
        print('y (poses) shape: ',self.y_data.shape)

    def set_transform(self,transform):
        self.transform = transform

    def __len__(self):
        return self.y_data.shape[0]

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        if(self.transform):
            x = self.transform(x)

        return x,y


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
