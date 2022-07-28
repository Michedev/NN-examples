from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

class AdultModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(67, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
            nn.Sigmoid()
        )
        self.lossf = nn.BCELoss()
        self.iteration = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        lossvalue = self.lossf(y_pred, y)
        if self.iteration % 5 == 0:
            self.log('train_loss', lossvalue)
        self.iteration += 1
        return dict(loss=lossvalue)

    def training_epoch_end(self, outputs):
        avg_loss = torch.FloatTensor([o['loss'] for o in outputs]).mean()
        self.log('test_avg_loss', avg_loss)

    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        X, y = batch
        y_pred = self(X)
        lossvalue = self.lossf(y_pred, y)
        y_pred_label = y_pred > 0.5
        accuracy = (y == y_pred_label).float().mean()
        self.log('test_loss', lossvalue)
        self.log('test_accuracy', accuracy)
        return dict(loss=lossvalue, accuracy=accuracy)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.FloatTensor([o['loss'] for o in outputs]).mean()
        avg_accuracy = torch.FloatTensor([o['accuracy'] for o in outputs]).mean()
        self.log('test_avg_loss', avg_loss)
        self.log('test_avg_accuracy', avg_accuracy)



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def main():
    dataset = pd.read_csv('data_preprocessed.csv')
    num_columns = ['age', 'fnlwgt', 'capital-gain','capital-loss','hours-per-week']
    for c in num_columns:
        dataset[c] = (dataset[c] - dataset[c].mean()) / (dataset[c].std()+1e-5)
    X, y = dataset.drop(columns='income_>50K'), dataset['income_>50K']
    # X = torch.from_numpy(X.values).float()
    # y = torch.from_numpy(y.values).float().unsqueeze(-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    rf = LogisticRegression()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    print('accuracy test:', accuracy_test)
    # train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)
    #
    # train_dl = DataLoader(train_dataset, batch_size=16)
    # test_dl = DataLoader(test_dataset, batch_size=16)
    #
    # trainer = pl.Trainer(max_epochs=10)
    #
    # model = AdultModel()
    #
    # trainer.fit(model, train_dl, test_dl)

if __name__ == '__main__':
    main()