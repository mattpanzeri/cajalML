from src.data_loaders.LickPrediction import NeuronsFiringRate
from src.data_loaders.MouseData import MouseData
from src.models import LinReg
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import trange
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from datetime import datetime

def train_session(session, h):
    mouse_data = MouseData(session)
    brain_areas = np.copy(mouse_data.recorded_brain_areas)
    del mouse_data
    for area in brain_areas:
        if area == 'root':
            continue
        print(f"loading {area} dataset from session {session}", flush=True)
        # create datasets
        train_idx, test_idx = train_test_split(np.arange(h['n_trials']), test_size=0.2, shuffle=True)
        train_ds = NeuronsFiringRate(session, trials=train_idx, areas=['MOp'], window=h['window'], offset=h['offset'], as_tensor=True)
        test_ds = NeuronsFiringRate(session, trials=test_idx, areas=['MOp'], window=h['window'], offset=h['offset'], as_tensor=True)
        print("...done", flush=True)
        weights = torch.tensor([len(train_ds.y.flatten())*h['window']/train_ds.TP], dtype=torch.float32)

        # create dataloaders
        train_dl = DataLoader(train_ds, batch_size=h['batch_size'], shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=h['batch_size'], shuffle=False)

        # initialize model, loss and optimizer
        model = LinReg.LinReg(train_ds.n_neurons.sum()*h['window'], 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=h['lr'])

        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

        # initialize logger
        now = datetime.now()
        date_time = now.strftime("%Y%m%d-%H%M%S")
        log_dir = f"runs/session_{session}/{area}_window_{h['window']}_batch_{h['batch_size']}_epochs_{h['epochs']}_{date_time}"
        logger = SummaryWriter(log_dir)
        train_model(model, train_dl, test_dl, h['epochs'], criterion, optimizer, logger)


def log_epoch(logger, epoch, losses, matrices, auc, f1, prefix):
    logger.add_scalar(prefix + '/mean_loss', np.mean(losses), epoch)
    logger.add_scalar(prefix + '/std_loss', np.std(losses), epoch)
    logger.add_scalar(prefix + '/TN_mean', np.mean(matrices[:, 0, 0]), epoch)
    logger.add_scalar(prefix + '/TN_std', np.std(matrices[:, 0, 0]), epoch)
    logger.add_scalar(prefix + '/FP_mean', np.mean(matrices[:, 0, 1]), epoch)
    logger.add_scalar(prefix + '/FP_std', np.std(matrices[:, 0, 1]), epoch)
    logger.add_scalar(prefix + '/FN_mean', np.mean(matrices[:, 1, 0]), epoch)
    logger.add_scalar(prefix + '/FN_std', np.std(matrices[:, 1, 0]), epoch)
    logger.add_scalar(prefix + '/TP_mean', np.mean(matrices[:, 1, 1]), epoch)
    logger.add_scalar(prefix + '/TP_std', np.std(matrices[:, 1, 1]), epoch)
    logger.add_scalar(prefix + '/AUC_mean', np.mean(auc), epoch)
    logger.add_scalar(prefix + '/AUC_std', np.std(auc), epoch)
    logger.add_scalar(prefix + '/F1_mean', np.mean(f1), epoch)
    logger.add_scalar(prefix + '/F1_std', np.std(f1), epoch)
    conf_matrix = np.mean(matrices, axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.tight_layout()
    logger.add_figure(prefix + '/confusion_matrix', fig, epoch)
    plt.close('all')
    


def train_model(model, train_dl, test_dl, n_epochs, criterion, optimizer, logger):
    # train the model
    for epoch in trange(n_epochs, desc='training model'):
        # train the model
        losses = []
        confusion_matrices = []
        auc = []
        f1 = []
        for X, y in train_dl:
            optimizer.zero_grad()
            ypred = model.forward(X)
            loss = criterion(ypred, y)
            loss.backward()
            optimizer.step()
            y_sig = (torch.sigmoid(ypred).detach().numpy() > 0.5).astype(int)
            y_det = y.detach().numpy().astype(int)
            losses.append(loss.item())
            confusion_matrices.append(confusion_matrix(y_det, y_sig, labels=[0, 1], normalize='true'))
            try:
                auc.append(roc_auc_score(y_det, (torch.sigmoid(ypred).detach().numpy()), labels=[0,1]))
            except ValueError:
                pass
            f1.append(f1_score(y_det, y_sig))

        # log the epoch
        log_epoch(logger, epoch, losses, np.array(confusion_matrices), np.array(auc), np.array(f1), 'train')

        # evaluate the model
        losses = []
        confusion_matrices = []
        auc = []
        f1 = []
        for X, y in test_dl:
            with torch.no_grad():
                ypred = model.forward(X)
                loss = criterion(ypred, y)
                losses.append(loss.item())
                y_sig = (torch.sigmoid(ypred).detach().numpy() > 0.5).astype(int)
                y_det = y.detach().numpy().astype(int)
                confusion_matrices.append(confusion_matrix(y_det, y_sig, labels=[0, 1], normalize='true'))
                try:
                    auc.append(roc_auc_score(y_det, (torch.sigmoid(ypred).detach().numpy()), labels=[0,1]))
                except ValueError:
                    pass
                f1.append(f1_score(y_det, y_sig))
        # log the epoch
        log_epoch(logger, epoch, losses, np.array(confusion_matrices), np.array(auc), np.array(f1), 'test')

if __name__ == '__main__':
    # params
    hparams = dict(
        n_trials = 100,
        window = 1,
        offset = 0,
        epochs = 200,
        batch_size = 2000,
        lr = 5e-5
    )

    #train_session(20, hparams)
    # train the model
    for i in range(39):
        train_session(i, hparams)