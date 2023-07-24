from src.data_loaders.MeanFiring import MeanFiringSingleArea
from src.models.LinReg import LinReg
import torch
import torch.nn as nn
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

model = LinReg(1, 1)
critereon = nn.MSELoss()
generator = torch.Generator().manual_seed(42)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 100
batch_size = 100
losses = []

dataset = MeanFiringSingleArea(10, 'MOp', absolute=True)
# split the dataset into train and test
train_set, test_set = train_test_split(dataset, test_size=0.2, shuffle=False)
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_set, batch_size=batch_size)

loss_train = []
loss_test = []
for epoch in tqdm.trange(epochs, desc='training model'):
    # train the model
    lt = []
    for x, y in train_data:
        ypred = model.forward(x)
        loss = critereon(ypred, y)
        lt.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train.append(np.mean(lt))
    # evaluate the model
    lt = []
    with torch.no_grad():
        for x, y in test_data:
            ypred = model.forward(x)
            loss = critereon(ypred, y)
            lt.append(loss.item())
    loss_test.append(np.mean(lt))
    

plt.figure()
plt.plot(loss_train, label='train')
plt.plot(loss_test, label='test')
plt.legend()
plt.show()