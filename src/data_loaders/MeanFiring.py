from src.data_loaders.MouseData import MouseData
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch

class MeanFiringSingleArea(MouseData, Dataset):

    def __init__(self, mouse, area, absolute=False):
        super().__init__(mouse)

        # compute mean firing rate across neurons in the area
        self.area = area
        self.absolute = absolute
        X = self.spikes[self.brain_areas == area, :, :].mean(axis=0) / self.dt
        self.X = torch.tensor(X.reshape((self.NTr*self.NTi,1)), dtype=torch.float32)
        # target is the wheel movement
        self.y = torch.tensor(self.wheel.reshape((self.NTr*self.NTi,1)), dtype=torch.float32)
        if self.absolute:
            self.y = torch.abs(self.y)

    def __len__(self):
        return self.NTr*self.NTi
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    data = MeanFiringSingleArea(10, 'MOp', absolute=True)
    
    plt.figure()
    plt.plot(data.X)
    plt.plot(data.y)
    plt.show()