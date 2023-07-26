from torch.utils.data import Dataset
from src.data_loaders.MouseData import MouseData
import torch
import numpy as np
import matplotlib.pyplot as plt

class MeanFiringPerArea(MouseData, Dataset):
    """
    X is the mean firing rate across neurons, per area
    there is no time window component
    """
    def __init__(self, session, trials=None):
        """
        Parameters
        ----------
        session : int
            the session to use
        trials : array of ints
            the trials to use for the dataset
        """
        super().__init__(session)

        # get mean spiking per brain region
        data = []
        self.n_areas = len(self.recorded_brain_areas)
        for area in self.recorded_brain_areas:
            X = self.spikes[self.brain_areas == area, :, :].mean(axis=0) / self.dt # average across neurons
            data.append(X[trials, :]) # select trials
        self.X = torch.tensor(np.array(data).reshape(self.n_areas,-1).T, dtype=torch.float32)
        self.y = torch.tensor(self.licks.squeeze()[trials].reshape(-1,1), dtype=torch.float32)

    def __len__(self):
        return self.NTr*self.NTi
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def plot(self):
        plt.figure()
        plt.imshow(self.X.T, aspect='auto', vmin=0, vmax=20, cmap='jet')
        plt.colorbar(label='mean firing rate (Hz)')
        plt.yticks(np.arange(self.n_areas), self.recorded_brain_areas)
        plt.plot(self.n_areas-self.y-0.5, 'k')
        plt.show()

class AllNeuronsPerArea(MouseData, Dataset):
    """
    X is the firing rates for all neurons, sorted by area
    there is no time window component
    """
    def __init__(self, session, trials=None):
        """
        Parameters
        ----------
        session : int
            the session to use
        trials : array of ints
            the trials to use for the dataset
        """
        super().__init__(session)

        # get data ordered by brain region
        data = []
        n_neurons = []
        self.n_areas = len(self.recorded_brain_areas)
        for area in self.recorded_brain_areas:
            X = self.spikes[self.brain_areas == area, :, :] / self.dt
            n_neurons.append(X.shape[0])
            data.append(X[:,trials])
        self.n_neurons = np.array(n_neurons)
        self.X = torch.tensor(np.vstack(data).reshape(self.n_neurons.sum(),-1).T, dtype=torch.float32)
        self.y = torch.tensor(self.licks.squeeze()[trials].reshape(-1,1), dtype=torch.float32)

    def __len__(self):
        return self.NTr*self.NTi
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


if __name__ == "__main__":
    ds = AllNeuronsPerArea(10, np.arange(50))
    print(ds.X.shape)
    print(ds.y.shape)