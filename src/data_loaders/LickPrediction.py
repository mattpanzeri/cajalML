from torch.utils.data import Dataset
from src.data_loaders.MouseData import MouseData
import torch
import numpy as np
import matplotlib.pyplot as plt

class MeanFiringRate(MouseData, Dataset):

    def __init__(self, session, trials=None, areas=None, window=1, offset=0, as_tensor=True):
        # initialize the dataset object
        super().__init__(session)
        self.session = session
        self.trials = trials
        self.window = window
        self.offset = offset
        self.as_tensor = as_tensor
        self.areas = areas if areas is not None else self.recorded_brain_areas
        self.n_areas = len(self.areas)
        # get mean spiking per brain region
        data = []
        for area in self.areas:
            X = self.spikes[self.brain_areas == area, :, :].mean(axis=0) / self.dt # average across neurons
            data.append(X[trials, :]) # select trials
        self.X = np.array(data) # shape is (n_areas, n_trials, n_time)
        self.n_trials = self.X.shape[1]
        self.y = self.licks.squeeze()[trials]
        self.n_time = self.NTi - self.window - self.offset
        
    def __len__(self):
        return self.n_trials * self.n_time

    def __getitem__(self, index):
        trial = index // self.n_trials
        time = (index % self.n_time) 
        start, end = time, time + self.window + self.offset
        X = self.X[:, trial, start: end].flatten()
        y = self.y[trial, end].flatten()
        if self.as_tensor:
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            return X, y
        

class NeuronsFiringRate(MouseData, Dataset):

    def __init__(self, session, trials=None, areas=None, window=1, offset=0, as_tensor=True):
        # initialize the dataset object
        super().__init__(session)
        self.session = session
        self.trials = trials
        self.window = window
        self.offset = offset
        self.as_tensor = as_tensor
        self.areas = areas if areas is not None else self.recorded_brain_areas
        self.n_areas = len(self.areas)
        # get mean spiking per brain region
        data = []
        n_neurons = []
        for area in self.areas:
            X = self.spikes[self.brain_areas == area, :, :] / self.dt
            n_neurons.append(X.shape[0])
            data.append(X[:,trials])
        self.n_neurons = np.array(n_neurons)
        self.X = np.vstack(data) # shape is (n_areas, n_trials, n_time)
        self.n_trials = self.X.shape[1]
        self.y = self.licks.squeeze()[trials]
        self.TN = (self.y==0).sum()
        self.TP = (self.y==1).sum()
        self.n_time = self.NTi - self.window - self.offset 
        
    def __len__(self):
        return self.n_trials * self.n_time

    def __getitem__(self, index):
        trial = index // self.n_time
        time = index % self.n_time
        start, end = time - self.offset, time + self.window - self.offset
        X = self.X[:, trial, start: end].flatten()
        y = self.y[trial, end].flatten()
        if self.as_tensor:
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            return X, y

if __name__ == "__main__":
    ds = NeuronsFiringRate(10, trials = np.arange(50), window=1, offset = 0, areas=["MOp"])
    print(ds[0][0].shape, ds[0][1].shape, len(ds), ds.n_neurons)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=500, shuffle=False)
    x, y = next(iter(dl))
    print(x.shape, y.shape)

    plt.figure()
    plt.imshow(x.T, aspect='auto', vmin=0, interpolation='none', cmap='jet')
    plt.colorbar()
    plt.show()