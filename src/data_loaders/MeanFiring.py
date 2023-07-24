from src.data_loaders.MouseData import MouseData
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MeanFiringSingleArea(MouseData, Dataset):

    def __init__(self, mouse, area):
        super().__init__(mouse)

        # compute mean firing rate across neurons in the area
        self.area = area
        X = self.spikes[self.brain_areas == area, :, :].mean(axis=0) / self.dt
        self.X = X.reshape((self.NTr*self.NTi,1))
        # target is the wheel movement
        self.y = self.wheel.reshape((self.NTr*self.NTi,1))

    def __len__(self):
        return self.NTr*self.NTi
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    data = MeanFiringSingleArea(10, 'MOp')
    
    plt.figure()
    plt.plot(data.X)
    plt.plot(data.y)
    plt.show()